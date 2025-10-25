import json
import sys
from typing import Dict, List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QApplication,
)


class AnnotationReviewDialog(QDialog):

    def __init__(
        self,
        frame: QImage,
        detections: List[Dict],
        class_choices: List[Tuple[int, str]],
        parent: QWidget = None,
        
    ):
        super().__init__(parent)
        self.setWindowTitle("Annotations for Retraining Data")
        self.resize(900, 600)
        self.setWindowIcon(QIcon("res/PolyVisionLogo.png"))
        self.setModal(True)

        self._class_choices = class_choices
        self._detections = [
            {
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "class_id": det.get("class_id", class_choices[0][0] if class_choices else 0),
                "score": det.get("score", 0.0),
                "meta": det.get("meta", {}),
                "accepted": True,
            }
            for det in detections
        ]

        self._source_image = frame.copy()
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumHeight(360)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Keep", "Class", "Score", "Bounding Box", "Info"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.TablePopulation()
        self._table.resizeColumnsToContents()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Confirm which detections should be added to the retraining dataset. "
            "Unchecked boxes will be ignored."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        layout.addWidget(self._preview_label)
        layout.addWidget(self._table)
        layout.addWidget(buttons)

        self.updatePreview()

    def TablePopulation(self) -> None:
        self._table.setRowCount(len(self._detections))
        for row, det in enumerate(self._detections):
            # Keep checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(det["accepted"])
            checkbox.stateChanged.connect(self._make_keep_handler(row))
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(checkbox)
            self._table.setCellWidget(row, 0, container)

            # Class combo
            combo = QComboBox()
            for value, label in self._class_choices:
                combo.addItem(label, value)
            current_index = combo.findData(det["class_id"])
            if current_index == -1:
                current_index = 0
            combo.setCurrentIndex(current_index)
            combo.currentIndexChanged.connect(self.classHandler(row, combo))
            self._table.setCellWidget(row, 1, combo)

            # Score (read only)
            score_item = QTableWidgetItem(f"{det['score'] * 100:.2f}%")
            score_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 2, score_item)

            # BBox coordinates
            bbox = det["bbox"]
            bbox_text = ", ".join(f"{int(coord)}" for coord in bbox)
            bbox_item = QTableWidgetItem(bbox_text)
            bbox_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 3, bbox_item)

            # Meta info
            meta_text = json.dumps(det["meta"]) if det["meta"] else "-"
            meta_item = QTableWidgetItem(meta_text)
            meta_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._table.setItem(row, 4, meta_item)

    def _make_keep_handler(self, index: int):
        def handler(state: int) -> None:
            self._detections[index]["accepted"] = state == Qt.Checked
            self.updatePreview()

        return handler

    def classHandler(self, index: int, combo: QComboBox):
        def handler(_: int) -> None:
            self._detections[index]["class_id"] = combo.currentData()
            self.updatePreview()

        return handler

    def updatePreview(self) -> None:
        pixmap = QPixmap.fromImage(self._source_image)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        for det in self._detections:
            bbox = det["bbox"]
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            accepted = det.get("accepted", False)
            color = QColor(0, 200, 0) if accepted else QColor(200, 0, 0)
            pen = QPen(color, 2 if accepted else 1, Qt.SolidLine if accepted else Qt.DotLine)
            painter.setPen(pen)
            painter.drawRect(*rect)

            class_id = det.get("class_id")
            class_label = next((label for value, label in self._class_choices if value == class_id), str(class_id))
            score = det.get("score", 0.0)
            text = f"{class_label} ({score * 100:.1f}%)"
            painter.drawText(rect[0], rect[1] - 4, text)

        painter.end()

        scaled = pixmap.scaled(
            self._preview_label.width() if self._preview_label.width() > 0 else pixmap.width(),
            self._preview_label.height() if self._preview_label.height() > 0 else pixmap.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updatePreview()

    def _on_accept(self) -> None:
        if not any(det.get("accepted") for det in self._detections):
            reply = QMessageBox.question(
                self,
                "Save as Negative?",
                "All detections are marked as rejected. Save this frame as a negative sample?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
        self.accept()

    def accepted_annotations(self) -> List[Dict]:
        result = []
        for det in self._detections:
            entry = dict(det)
            entry.pop("meta", None)
            if entry.get("accepted"):
                entry["review_status"] = "accepted"
                result.append(entry)
        return result

    def rejected_annotations(self) -> List[Dict]:
        result = []
        for det in self._detections:
            entry = dict(det)
            entry.pop("meta", None)
            if not entry.get("accepted"):
                entry["review_status"] = "rejected"
                result.append(entry)
        return result

def main():
    app = QApplication(sys.argv)

    width, height = 640, 640
    sample_image = QImage(width, height, QImage.Format_RGB32)
    sample_image.fill(QColor(245, 245, 245))

    sample_detections = [
        {"bbox": [80, 60, 260, 260], "class_id": 1, "score": 0.88},
        {"bbox": [300, 280, 480, 520], "class_id": 3, "score": 0.74},
    ]
    sample_classes = [(1, "Filament"), (2, "Film"), (3, "Fragment")]

    dialog = AnnotationReviewDialog(sample_image, sample_detections, sample_classes)
    dialog.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
