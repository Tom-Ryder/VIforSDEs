from __future__ import annotations

import time
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Iterator

import torch
from rich.console import Console as RichConsole
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from variational_sde.config import TrainingConfig
    from variational_sde.inference.types import EvidenceLowerBoundComponents
    from variational_sde.posterior.variational_posterior import (
        InferenceDiagnostics,
        VariationalPosteriorSummary,
    )


def _grid(*cols: str) -> Table:
    t = Table.grid(padding=(0, 2))
    for style in cols:
        t.add_column(style=style)
    return t


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m"


class TrainingProgress:
    def __init__(
        self,
        console: Console,
        n_iterations: int,
        update_interval: int = 10,
        param_names: Sequence[str] | None = None,
    ) -> None:
        self._console = console
        self._n_iterations = n_iterations
        self._update_interval = update_interval
        self._param_names = param_names
        self._start_time = 0.0
        self._last_step = 0
        self._last_time = 0.0
        self._iter_per_sec = 0.0
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._step = 0
        self._loss = 0.0
        self._elbo = 0.0
        self._best_elbo = 0.0
        self._components: EvidenceLowerBoundComponents | None = None
        self._grad_norm: float | None = None
        self._param_means: torch.Tensor | None = None

    def __enter__(self) -> TrainingProgress:
        if not self._console.enabled:
            return self
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._progress = Progress(
            TextColumn("[cyan]VI-SDE"),
            BarColumn(bar_width=40, style="dim", complete_style="cyan"),
            TextColumn("[dim]{task.percentage:>5.1f}%"),
            TextColumn("[dim]{task.completed:>6}/{task.total}"),
            console=self._console._console,
            expand=False,
        )
        self._task_id = self._progress.add_task("training", total=self._n_iterations)
        self._live = Live(
            self._build_display(),
            console=self._console._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._step > 0:
            self._console._console.print(self._build_completion())

    def update(
        self,
        step: int,
        loss: float,
        elbo: float,
        best_elbo: float,
        components: EvidenceLowerBoundComponents | None = None,
        grad_norm: float | None = None,
        param_means: torch.Tensor | None = None,
    ) -> None:
        now = time.perf_counter()
        if step > self._last_step:
            dt = now - self._last_time
            if dt > 0:
                instant_rate = (step - self._last_step) / dt
                self._iter_per_sec = 0.9 * self._iter_per_sec + 0.1 * instant_rate
            self._last_step = step
            self._last_time = now

        self._step, self._loss, self._elbo, self._best_elbo = (
            step,
            loss,
            elbo,
            best_elbo,
        )
        self._components, self._grad_norm, self._param_means = (
            components,
            grad_norm,
            param_means,
        )

        if not self._console.enabled:
            return
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=step + 1)
        if self._live is not None and step % self._update_interval == 0:
            self._live.update(self._build_display())

    def _build_display(self) -> Panel:
        parts: list[Text | Progress | Table] = []
        if self._progress:
            parts.append(self._progress)

        elapsed = time.perf_counter() - self._start_time
        remaining = (self._n_iterations - self._step - 1) / max(
            self._iter_per_sec, 0.01
        )

        timing = _grid("dim", "bold", "dim", "bold", "dim", "bold")
        timing.add_row(
            "Elapsed",
            _format_time(elapsed),
            "ETA",
            _format_time(remaining) if self._iter_per_sec > 0 else "--",
            "Speed",
            f"{self._iter_per_sec:.1f} it/s" if self._iter_per_sec > 0 else "--",
        )
        parts.append(timing)

        if self._step > 0:
            parts.append(Text())
            metrics = _grid("dim", "bold", "dim", "bold")
            metrics.add_row(
                "Loss", f"{self._loss:>12.4f}", "Best ELBO", f"{self._best_elbo:>12.4f}"
            )
            metrics.add_row(
                "ELBO",
                f"{self._elbo:>12.4f}",
                "Grad Norm",
                f"{self._grad_norm:>12.4f}" if self._grad_norm else "",
            )
            parts.append(metrics)

            if self._param_means is not None:
                parts.extend([Text(), Text("Parameter Means", style="dim")])
                params = _grid("dim", "cyan")
                means = self._param_means.detach().cpu()
                names = (
                    list(self._param_names)
                    if self._param_names
                    else [f"θ[{i}]" for i in range(means.shape[0])]
                )
                for i, name in enumerate(names):
                    params.add_row(f"  {name}", f"{means[i].item():.4f}")
                parts.append(params)

            if self._components:
                parts.extend([Text(), Text("ELBO Components", style="dim")])
                comp = _grid("dim", "")
                c = self._components
                for label, val in [
                    ("Observation", c.observation_log_prob),
                    ("SDE", c.sde_log_prob),
                    ("Generative", c.generative_log_prob),
                    ("Prior", c.prior_log_prob),
                    ("Posterior", c.posterior_log_prob),
                ]:
                    comp.add_row(f"  {label}", f"{val.item():+.2f}")
                parts.append(comp)

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                parts.extend([Text(), Text(f"Memory: {mem:.2f} GB", style="dim")])

        return Panel(
            Group(*parts),
            title="[cyan]VI-SDE Training",
            border_style="dim",
            padding=(0, 1),
        )

    def _build_completion(self) -> Panel:
        elapsed = time.perf_counter() - self._start_time
        avg_speed = (self._step + 1) / elapsed if elapsed > 0 else 0
        t = _grid("dim", "bold", "dim", "bold")
        t.add_row("Iterations", f"{self._step + 1:,}", "Time", _format_time(elapsed))
        t.add_row("Final ELBO", f"{self._elbo:.4f}", "Speed", f"{avg_speed:.1f} it/s")
        return Panel(t, title="[green]Training Complete", border_style="green")


class PretrainProgress:
    def __init__(self, console: Console, max_iterations: int) -> None:
        self._console = console
        self._max_iterations = max_iterations
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._live: Live | None = None
        self._step = 0
        self._mse = 0.0
        self._best_mse = float("inf")
        self._sigma = 1.0

    def __enter__(self) -> PretrainProgress:
        if not self._console.enabled:
            return self
        self._progress = Progress(
            TextColumn("[yellow]Pretrain"),
            TextColumn("[dim]{task.completed:>5}"),
            TextColumn("[dim]σ:"),
            TextColumn("{task.fields[sigma]:>6.3f}"),
            TextColumn("[dim]MSE:"),
            TextColumn("{task.fields[mse]:>10.2f}"),
            TextColumn("[dim]Best:"),
            TextColumn("{task.fields[best]:>10.2f}"),
            console=self._console._console,
            expand=False,
        )
        self._task_id = self._progress.add_task(
            "pretrain", total=None, mse=0.0, best=float("inf"), sigma=1.0
        )
        self._live = Live(self._progress, console=self._console._console, transient=True)
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self._console.enabled:
            self._console._console.print(
                f"[yellow]Pretrain complete[/] · {self._step + 1} steps · σ: {self._sigma:.3f} · MSE: {self._best_mse:.2f}"
            )

    def update(self, step: int, mse: float, best_mse: float, sigma: float) -> None:
        self._step = step
        self._mse = mse
        self._best_mse = best_mse
        self._sigma = sigma
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id, completed=step + 1, mse=mse, best=best_mse, sigma=sigma
            )


class Console:
    def __init__(self, enabled: bool = True) -> None:
        self._console = RichConsole(quiet=not enabled)
        self.enabled = enabled

    @contextmanager
    def pretrain_progress(self, n_iterations: int) -> Iterator[PretrainProgress]:
        progress = PretrainProgress(self, n_iterations)
        with progress:
            yield progress

    @contextmanager
    def training_progress(
        self,
        n_iterations: int,
        update_interval: int = 10,
        param_names: Sequence[str] | None = None,
    ) -> Iterator[TrainingProgress]:
        progress = TrainingProgress(self, n_iterations, update_interval, param_names)
        with progress:
            yield progress

    def config_panel(self, config: TrainingConfig) -> None:
        if not self.enabled:
            return
        t = _grid("dim", "bold")
        t.add_row("Iterations", f"{config.n_iterations:,}")
        t.add_row("Batch size", f"{config.batch_size}")
        t.add_row("Time step", f"{config.time_step}")
        t.add_row("Learning rate", f"{config.learning_rate:.0e}")
        t.add_row("Grad clip", f"{config.grad_clip_norm}")
        self._console.print(
            Panel(t, title="[cyan]Configuration", border_style="dim", padding=(0, 1))
        )

    def summary_table(
        self,
        summary: VariationalPosteriorSummary,
        diagnostics: InferenceDiagnostics,
        param_names: Sequence[str] | None = None,
    ) -> None:
        if not self.enabled:
            return

        diag = _grid("dim", "bold")
        diag.add_row("Iterations", f"{diagnostics.n_iterations:,}")
        diag.add_row("Final ELBO", f"{diagnostics.final_evidence_lower_bound:.4f}")
        self._console.print(
            Panel(
                diag,
                title="[green]Training Complete",
                border_style="green",
                padding=(0, 1),
            )
        )

        n_params = summary.sde_parameter_mean.shape[0]
        names = (
            list(param_names) if param_names else [f"θ[{i}]" for i in range(n_params)]
        )

        t = Table(
            title="[cyan]Parameter Posterior",
            border_style="dim",
            header_style="bold cyan",
        )
        t.add_column("Parameter", style="bold")
        t.add_column("Mean", justify="right")
        t.add_column("Std", justify="right")
        t.add_column("95% CI", justify="right", style="dim")

        mean, std = summary.sde_parameter_mean.cpu(), summary.sde_parameter_std.cpu()
        q05, q95 = (
            summary.sde_parameter_quantiles.q05.cpu(),
            summary.sde_parameter_quantiles.q95.cpu(),
        )

        for i, name in enumerate(names):
            t.add_row(
                name,
                f"{mean[i].item():.4f}",
                f"{std[i].item():.4f}",
                f"[{q05[i].item():.4f}, {q95[i].item():.4f}]",
            )

        self._console.print(t)

    def print(self, message: str, style: str | None = None) -> None:
        if self.enabled:
            self._console.print(message, style=style)
