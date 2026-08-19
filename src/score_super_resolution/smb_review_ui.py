"""Interactive, outcome-blind human review UI for the locked SMB audit."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output

from score_super_resolution.benchmark_policy import BenchmarkPurpose
from score_super_resolution.review_evidence import (
    canonical_review_csv,
    read_review,
    validate_human_cell,
)
from score_super_resolution.smb import load_smb

PAGE_SUFFIX = re.compile(r"_p\d+$")
BATCH_SIZE = 16
INDIVIDUAL_REVIEW_MARKER = "[individual-review]"
EXCEPTION_IDS = {
    "smb-test-000146",
    "smb-test-000284",
    "smb-test-000539",
    "smb-test-000674",
}


class SMBReviewSession:
    """Preserve and progressively update the tracked human-review CSV."""

    def __init__(self, project_root: Path | str = ".", *, dataset: Any | None = None) -> None:
        start = Path(project_root).resolve()
        self.root = next(
            (
                candidate
                for candidate in (start, *start.parents)
                if (candidate / "pyproject.toml").is_file()
                and (candidate / "data/audits/smb-review-v1.csv").is_file()
            ),
            None,
        )
        if self.root is None:
            raise RuntimeError("Could not locate the proyecto/ root")
        self.review_path = self.root / "data/audits/smb-review-v1.csv"
        self.sample_path = self.root / "data/audits/smb-visual-sample-v1.csv"
        sample = pd.read_csv(self.sample_path, dtype=str)
        self.sample_ids = sample["item_id"].tolist()
        self.visual_item_ids = self.sample_ids + sorted(EXCEPTION_IDS - set(self.sample_ids))
        self.candidate_keys = [
            row["review_key"] for row in self.read_rows() if row["review_kind"] == "candidate"
        ]
        self.dataset = dataset or load_smb(purpose=BenchmarkPurpose.FIXED_VISUAL_AUDIT)
        if len(self.dataset) != 685:
            raise RuntimeError(f"Expected 685 SMB rows, got {len(self.dataset)}")
        self.reviewer = widgets.Text(description="Revisor:", placeholder="Nombre y apellidos")
        self.confirm_policy = widgets.Checkbox(
            value=False, description="Confirmo la política descrita"
        )

    def read_rows(self) -> list[dict[str, str]]:
        return list(read_review(self.review_path).rows)

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        """Atomically save without changing the header or row order."""
        content = canonical_review_csv(rows)
        temporary = self.review_path.with_suffix(".csv.tmp")
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.review_path)

    def summary(self) -> dict[str, int]:
        rows = self.read_rows()
        groups = {row["source_group_id"] for row in rows if row["review_kind"] == "item"}
        return {
            "rows": len(rows),
            "reviewed": sum(row["review_status"] == "reviewed" for row in rows),
            "pending": sum(row["review_status"] != "reviewed" for row in rows),
            "visual_items": len(self.visual_item_ids),
            "sample_items": len(self.sample_ids),
            "candidates": len(self.candidate_keys),
            "groups": len(groups),
        }

    def apply_policy(self, reviewer: str) -> dict[str, int]:
        """Apply the explicitly confirmed group/rights/sample policy to pending item rows."""
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("Indica el nombre del revisor")
        validate_human_cell(reviewer, field="reviewer", review_key="session-policy")
        rows = self.read_rows()
        visual_set = set(self.visual_item_ids)
        today = date.today().isoformat()
        changed = 0
        for row in rows:
            if row["review_kind"] != "item" or row["review_status"] == "reviewed":
                continue
            row["source_group_id"] = PAGE_SUFFIX.sub("", row["source_group_id"])
            row["quality_disposition"] = "acceptable"
            row["suitability_disposition"] = "suitable"
            row["duplicate_disposition"] = "distinct"
            row["dataset_licence_status"] = "confirmed"
            row["item_provenance_status"] = "unavailable"
            row["access_status"] = "confirmed"
            row["redistribution_status"] = "permitted"
            row["figure_reproduction_status"] = "permitted"
            if row["item_id"] not in visual_set:
                row["review_status"] = "reviewed"
                row["reviewer"] = reviewer
                row["reviewed_at"] = today
                row["rationale"] = (
                    "Revisión mediante auditoría automática completa y política muestral fijada; "
                    "página procesada sin incidencia técnica y fuera de la muestra visual. Uso no "
                    "comercial sujeto a CC BY-NC 4.0 y atribución."
                )
            changed += 1
        self.write_rows(rows)
        result = self.summary()
        result["changed"] = changed
        return result

    def save_item(
        self,
        *,
        item_id: str,
        reviewer: str,
        quality_flags: tuple[str, ...],
        suitability: str,
        rationale: str,
    ) -> None:
        reviewer = reviewer.strip()
        rationale = rationale.strip()
        if not reviewer or not rationale:
            raise ValueError("Revisor y justificación son obligatorios")
        validate_human_cell(reviewer, field="reviewer", review_key=item_id)
        validate_human_cell(rationale, field="rationale", review_key=item_id)
        rows = self.read_rows()
        row = next(row for row in rows if row["review_key"] == item_id)
        policy_fields = (
            "source_group_id",
            "dataset_licence_status",
            "item_provenance_status",
            "access_status",
            "redistribution_status",
            "figure_reproduction_status",
        )
        if any(not row[field] or row[field] == "pending" for field in policy_fields):
            raise ValueError("Aplica primero la política general")
        row["quality_disposition"] = (
            ";".join(sorted(set(quality_flags))) if quality_flags else "acceptable"
        )
        row["suitability_disposition"] = suitability
        row["review_status"] = "reviewed"
        row["reviewer"] = reviewer
        row["reviewed_at"] = date.today().isoformat()
        row["rationale"] = rationale
        self.write_rows(rows)

    def save_candidate(
        self,
        *,
        review_key: str,
        reviewer: str,
        disposition: str,
        rationale: str,
    ) -> None:
        reviewer = reviewer.strip()
        rationale = rationale.strip()
        if not reviewer or not rationale:
            raise ValueError("Revisor y justificación son obligatorios")
        validate_human_cell(reviewer, field="reviewer", review_key=review_key)
        validate_human_cell(rationale, field="rationale", review_key=review_key)
        rows = self.read_rows()
        row = next(row for row in rows if row["review_key"] == review_key)
        involved = {row["item_id"], row["candidate_item_id"]}
        conflicts = [
            other["review_key"]
            for other in rows
            if other["review_kind"] == "candidate"
            and other["review_status"] == "reviewed"
            and other["review_key"] != review_key
            and involved.intersection({other["item_id"], other["candidate_item_id"]})
            and other["duplicate_disposition"] != disposition
        ]
        if conflicts:
            raise ValueError(
                "Una página tiene relaciones incompatibles; informa a Codex: " + str(conflicts)
            )
        row["review_status"] = "reviewed"
        row["reviewer"] = reviewer
        row["reviewed_at"] = date.today().isoformat()
        row["rationale"] = rationale
        row["duplicate_disposition"] = disposition
        for item_id in involved:
            item = next(item for item in rows if item["review_key"] == item_id)
            item["duplicate_disposition"] = disposition
        self.write_rows(rows)

    def save_item_batch(
        self,
        *,
        item_ids: list[str],
        reviewer: str,
        excluded_ids: set[str],
        batch_number: int,
    ) -> dict[str, int]:
        """Approve a displayed batch while leaving declared anomalies for individual review."""
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("Indica el nombre del revisor")
        validate_human_cell(reviewer, field="reviewer", review_key=f"batch-{batch_number}")
        unknown = excluded_ids - set(item_ids)
        if unknown:
            raise ValueError(f"Las exclusiones no pertenecen al lote: {sorted(unknown)}")
        rows = self.read_rows()
        today = date.today().isoformat()
        approved = 0
        already_reviewed = 0
        flagged = 0
        for item_id in item_ids:
            row = next(row for row in rows if row["review_key"] == item_id)
            if item_id in excluded_ids:
                if row["review_status"] != "reviewed":
                    row["reviewer"] = reviewer
                    row["rationale"] = (
                        f"{INDIVIDUAL_REVIEW_MARKER} Marcada en el lote {batch_number}; "
                        "pendiente de ampliación y descripción."
                    )
                    flagged += 1
                continue
            if row["review_status"] == "reviewed":
                already_reviewed += 1
                continue
            policy_fields = (
                "source_group_id",
                "dataset_licence_status",
                "item_provenance_status",
                "access_status",
                "redistribution_status",
                "figure_reproduction_status",
            )
            if any(not row[field] or row[field] == "pending" for field in policy_fields):
                raise ValueError("Aplica primero la política general")
            row["quality_disposition"] = "acceptable"
            row["suitability_disposition"] = "suitable"
            row["review_status"] = "reviewed"
            row["reviewer"] = reviewer
            row["reviewed_at"] = today
            row["rationale"] = (
                "Revisión visual por lote de la muestra fijada: legibilidad, contraste y "
                "procesabilidad adecuados; no se observó una anomalía material."
            )
            approved += 1
        self.write_rows(rows)
        return {
            "approved": approved,
            "already_reviewed": already_reviewed,
            "excluded": len(excluded_ids),
            "flagged": flagged,
        }

    def flagged_item_ids(self) -> list[str]:
        """Return persisted, still-pending pages selected for individual review."""
        rows_by_key = {row["review_key"]: row for row in self.read_rows()}
        return [
            item_id
            for item_id in self.visual_item_ids
            if rows_by_key[item_id]["review_status"] != "reviewed"
            and rows_by_key[item_id]["rationale"].startswith(INDIVIDUAL_REVIEW_MARKER)
        ]

    def related_candidate_rows(self, review_key: str) -> list[dict[str, str]]:
        """Return every other candidate decision sharing either page with the current pair."""
        rows = self.read_rows()
        current = next(row for row in rows if row["review_key"] == review_key)
        involved = {current["item_id"], current["candidate_item_id"]}
        return [
            row
            for row in rows
            if row["review_kind"] == "candidate"
            and row["review_key"] != review_key
            and involved.intersection({row["item_id"], row["candidate_item_id"]})
        ]

    def _image(self, item_id: str) -> Any:
        return self.dataset[int(item_id.removeprefix("smb-test-"))]["image"]

    def policy_widget(self) -> widgets.Widget:
        button = widgets.Button(description="Aplicar política", button_style="warning")
        output = widgets.Output()

        def apply(_button: widgets.Button) -> None:
            with output:
                clear_output()
                try:
                    if not self.confirm_policy.value:
                        raise ValueError("Marca la confirmación antes de aplicar la política")
                    result = self.apply_policy(self.reviewer.value)
                    progress = f"{result['reviewed']}/{result['rows']}"
                    print(f"Política aplicada. {progress} filas; {result['groups']} grupos.")
                except Exception as error:
                    print(f"No se guardó: {error}")

        button.on_click(apply)
        return widgets.VBox([self.reviewer, self.confirm_policy, button, output])

    def batch_widget(self) -> widgets.Widget:
        batch_count = (len(self.visual_item_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        position = widgets.IntSlider(
            value=0,
            min=0,
            max=batch_count - 1,
            description="Lote:",
            continuous_update=False,
        )
        batch_ids = widgets.Textarea(
            description="IDs:",
            layout=widgets.Layout(width="900px", height="130px"),
        )
        exclusion_grid = widgets.GridBox(
            layout=widgets.Layout(
                grid_template_columns="repeat(4, minmax(210px, 1fr))",
                grid_gap="6px 12px",
                width="100%",
            )
        )
        confirm = widgets.Checkbox(
            value=False,
            description="He inspeccionado todas las páginas visibles",
        )
        show = widgets.Button(description="Mostrar lote", button_style="info")
        approve = widgets.Button(description="Aprobar lote", button_style="success")
        previous = widgets.Button(description="← Anterior")
        following = widgets.Button(description="Siguiente →")
        image_output = widgets.Output()
        status_output = widgets.Output()
        displayed_batch: int | None = None

        def current_ids() -> list[str]:
            start = position.value * BATCH_SIZE
            return self.visual_item_ids[start : start + BATCH_SIZE]

        def refresh_exclusion_grid() -> None:
            flagged = set(self.flagged_item_ids())
            exclusion_grid.children = tuple(
                widgets.Checkbox(
                    value=item_id in flagged,
                    description=item_id,
                    indent=False,
                    layout=widgets.Layout(width="auto"),
                )
                for item_id in current_ids()
            )

        def reset_confirmation(*_args: object) -> None:
            nonlocal displayed_batch
            displayed_batch = None
            confirm.value = False
            batch_ids.value = "\n".join(current_ids())
            refresh_exclusion_grid()
            with image_output:
                clear_output()
            with status_output:
                clear_output()
                print("Pulsa Mostrar lote antes de aprobarlo.")

        def render(_button: widgets.Button) -> None:
            nonlocal displayed_batch
            item_ids = current_ids()
            batch_ids.value = "\n".join(item_ids)
            refresh_exclusion_grid()
            rows_by_key = {row["review_key"]: row for row in self.read_rows()}
            with image_output:
                clear_output(wait=True)
                columns = 4
                row_count = (len(item_ids) + columns - 1) // columns
                figure, axes = plt.subplots(row_count, columns, figsize=(16, 5 * row_count))
                flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
                for axis, item_id in zip(flat_axes, item_ids, strict=False):
                    image = self._image(item_id)
                    axis.imshow(image)
                    axis.axis("off")
                    status = rows_by_key[item_id]["review_status"]
                    exception = " · EXCEPCIÓN" if item_id in EXCEPTION_IDS else ""
                    axis.set_title(f"{item_id}{exception}\n{status}", fontsize=9)
                for axis in flat_axes[len(item_ids) :]:
                    axis.axis("off")
                figure.suptitle(
                    f"Lote {position.value + 1}/{batch_count} · {len(item_ids)} páginas",
                    fontsize=14,
                )
                plt.tight_layout()
                plt.show()
            displayed_batch = position.value
            confirm.value = False
            with status_output:
                clear_output()
                print(
                    "Si alguna página requiere ampliación, marca su checkbox. "
                    "Después podrás revisarla individualmente."
                )

        def persist(_button: widgets.Button) -> None:
            with status_output:
                clear_output()
                try:
                    if displayed_batch != position.value:
                        raise ValueError("Muestra este lote antes de aprobarlo")
                    if not confirm.value:
                        raise ValueError("Confirma que has inspeccionado las páginas visibles")
                    excluded_ids = {
                        str(checkbox.description)
                        for checkbox in exclusion_grid.children
                        if isinstance(checkbox, widgets.Checkbox) and checkbox.value
                    }
                    result = self.save_item_batch(
                        item_ids=current_ids(),
                        reviewer=self.reviewer.value,
                        excluded_ids=excluded_ids,
                        batch_number=position.value + 1,
                    )
                    print(
                        f"Lote guardado: {result['approved']} aprobadas, "
                        f"{result['already_reviewed']} ya revisadas, "
                        f"{result['flagged']} marcadas para revisión individual."
                    )
                    flagged = self.flagged_item_ids()
                    refresh_exclusion_grid()
                    print("Cola individual guardada:", ", ".join(flagged) or "vacía")
                    print("Usa Siguiente para continuar con el próximo lote.")
                except Exception as error:
                    print(f"No se guardó: {error}")

        position.observe(reset_confirmation, names="value")
        show.on_click(render)
        approve.on_click(persist)
        previous.on_click(lambda _button: setattr(position, "value", max(0, position.value - 1)))
        following.on_click(
            lambda _button: setattr(position, "value", min(position.max, position.value + 1))
        )
        reset_confirmation()
        return widgets.VBox(
            [
                position,
                widgets.HBox([previous, show, following]),
                batch_ids,
                image_output,
                exclusion_grid,
                confirm,
                approve,
                status_output,
            ]
        )

    def item_widget(self) -> widgets.Widget:
        item_selector = widgets.Dropdown(options=[], description="Marcada:")
        quality = {
            flag: widgets.Checkbox(value=False, description=flag)
            for flag in ("blurred", "low_contrast", "oversized", "skewed", "unprocessable")
        }
        suitability = widgets.Dropdown(
            options=["suitable", "unsuitable", "unavailable"], description="Idoneidad:"
        )
        rationale = widgets.Textarea(
            description="Justificación:", layout=widgets.Layout(width="900px", height="80px")
        )
        update_queue = widgets.Button(description="Actualizar cola", button_style="info")
        show = widgets.Button(description="Mostrar página", button_style="info")
        save = widgets.Button(description="Guardar página", button_style="success")
        image_output = widgets.Output()
        queue_output = widgets.Output()
        status_output = widgets.Output()

        def current_item_id() -> str:
            if item_selector.value is None:
                raise ValueError("No hay páginas marcadas para revisión individual")
            return str(item_selector.value)

        def refresh_queue(_button: widgets.Button | None = None) -> None:
            flagged = self.flagged_item_ids()
            previous_value = item_selector.value
            item_selector.options = flagged
            if previous_value in flagged:
                item_selector.value = previous_value
            with queue_output:
                clear_output()
                print(f"Cola individual ({len(flagged)}):", ", ".join(flagged) or "vacía")

        def refresh(_button: widgets.Button) -> None:
            try:
                item_id = current_item_id()
            except ValueError as error:
                with status_output:
                    clear_output()
                    print(error)
                return
            row = next(row for row in self.read_rows() if row["review_key"] == item_id)
            image = self._image(item_id)
            with image_output:
                clear_output(wait=True)
                _figure, axis = plt.subplots(figsize=(10, 13))
                axis.imshow(image)
                axis.axis("off")
                exception = " · excepción" if item_id in EXCEPTION_IDS else ""
                axis.set_title(
                    f"{item_id}{exception}\n"
                    f"{PAGE_SUFFIX.sub('', row['source_group_id'])} · {image.width}x{image.height}"
                )
                plt.show()
            flags = row["quality_disposition"].split(";")
            for flag, checkbox in quality.items():
                checkbox.value = flag in flags
            if row["suitability_disposition"] in suitability.options:
                suitability.value = row["suitability_disposition"]
            rationale.value = (
                "" if row["rationale"].startswith(INDIVIDUAL_REVIEW_MARKER) else row["rationale"]
            )
            with status_output:
                clear_output()
                print(f"Estado: {row['review_status']} · {row['rationale']}")

        def persist(_button: widgets.Button) -> None:
            with status_output:
                clear_output()
                try:
                    item_id = current_item_id()
                    self.save_item(
                        item_id=item_id,
                        reviewer=self.reviewer.value,
                        quality_flags=tuple(
                            flag for flag, checkbox in quality.items() if checkbox.value
                        ),
                        suitability=suitability.value,
                        rationale=rationale.value,
                    )
                    print(f"Guardado: {item_id}")
                    refresh_queue()
                except Exception as error:
                    print(f"No se guardó: {error}")

        update_queue.on_click(refresh_queue)
        show.on_click(refresh)
        save.on_click(persist)
        refresh_queue()
        return widgets.VBox(
            [
                widgets.HBox([item_selector, update_queue, show]),
                queue_output,
                image_output,
                widgets.HBox([widgets.VBox(list(quality.values())), suitability]),
                rationale,
                save,
                status_output,
            ]
        )

    def candidate_widget(self) -> widgets.Widget:
        position = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.candidate_keys) - 1,
            description="Par:",
            continuous_update=False,
        )
        disposition = widgets.Dropdown(
            options=["distinct", "related", "duplicate", "unavailable"],
            description="Relación:",
        )
        rationale = widgets.Textarea(
            description="Justificación:", layout=widgets.Layout(width="900px", height="80px")
        )
        previous = widgets.Button(description="← Anterior")
        following = widgets.Button(description="Siguiente →")
        show = widgets.Button(description="Mostrar par", button_style="info")
        save = widgets.Button(description="Guardar par", button_style="success")
        image_output = widgets.Output()
        history_output = widgets.Output()
        status_output = widgets.Output()

        def current() -> dict[str, str]:
            key = self.candidate_keys[position.value]
            return next(row for row in self.read_rows() if row["review_key"] == key)

        def refresh(*_args: object) -> None:
            row = current()
            images = [self._image(row["item_id"]), self._image(row["candidate_item_id"])]
            with image_output:
                clear_output(wait=True)
                figure, axes = plt.subplots(1, 2, figsize=(16, 11))
                for axis, image, item_id in zip(
                    axes,
                    images,
                    (row["item_id"], row["candidate_item_id"]),
                    strict=True,
                ):
                    axis.imshow(image)
                    axis.axis("off")
                    axis.set_title(f"{item_id} · {image.width}x{image.height}")
                figure.suptitle(
                    f"{position.value + 1}/{len(self.candidate_keys)} · {row['review_key']}"
                )
                plt.tight_layout()
                plt.show()
            if row["duplicate_disposition"] in disposition.options:
                disposition.value = row["duplicate_disposition"]
            rationale.value = row["rationale"]
            related = self.related_candidate_rows(row["review_key"])
            with history_output:
                clear_output()
                print("Otras comparaciones que comparten alguna de estas páginas:")
                if not related:
                    print("  Ninguna.")
                for previous_row in related:
                    label = previous_row["duplicate_disposition"] or "pending"
                    reason = previous_row["rationale"] or "sin revisar"
                    print(
                        f"  {previous_row['review_status']:>8} · "
                        f"{previous_row['item_id']} ↔ {previous_row['candidate_item_id']} · "
                        f"{label} · {reason}"
                    )
            with status_output:
                clear_output()
                print(f"Estado: {row['review_status']}")

        def persist(_button: widgets.Button) -> None:
            with status_output:
                clear_output()
                try:
                    row = current()
                    self.save_candidate(
                        review_key=row["review_key"],
                        reviewer=self.reviewer.value,
                        disposition=disposition.value,
                        rationale=rationale.value,
                    )
                    print(f"Guardado: {row['review_key']}")
                    if position.value < position.max:
                        position.value += 1
                    else:
                        refresh()
                except Exception as error:
                    print(f"No se guardó: {error}")

        position.observe(refresh, names="value")
        show.on_click(refresh)
        previous.on_click(lambda _button: setattr(position, "value", max(0, position.value - 1)))
        following.on_click(
            lambda _button: setattr(position, "value", min(position.max, position.value + 1))
        )
        save.on_click(persist)
        return widgets.VBox(
            [
                position,
                show,
                image_output,
                history_output,
                disposition,
                rationale,
                widgets.HBox([previous, save, following]),
                status_output,
            ]
        )

    def progress_widget(self) -> widgets.Widget:
        button = widgets.Button(description="Comprobar progreso", button_style="info")
        output = widgets.Output()

        def check(_button: widgets.Button) -> None:
            with output:
                clear_output()
                summary = self.summary()
                print(f"Filas revisadas: {summary['reviewed']}/{summary['rows']}")
                print(f"Pendientes: {summary['pending']}")
                print(f"Grupos actuales: {summary['groups']} (objetivo de la política: 260)")
                if not summary["pending"]:
                    print("Revisión completa. Comunica a Codex: revisión SMB lista.")

        button.on_click(check)
        return widgets.VBox([button, output])
