import json
import os
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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QApplication,
)

class ClickableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_callback = None

    def mousePressEvent(self, event):
        try:
            if callable(self.click_callback):
                self.click_callback(event.pos())
        except Exception:
            pass
        super().mousePressEvent(event)

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

        # Left: preview
        self._preview_label = ClickableLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        # make preview bigger with balanced size
        self._preview_label.setMinimumSize(450, 400)
        self._preview_label.setMaximumSize(500, 500)
        self._preview_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Right: full review table (no Info column)
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["No.", "Keep", "Class", "Score", "Bounding Box"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Apply custom styling for better selection visibility
        self._table.setStyleSheet("""
            QTableWidget {
                selection-background-color: #D9D9D9;  
                selection-color: #000000;              
            }
            QTableWidget::item:selected {
                background: #D9D9D9;
                color: #000000;
            }
        """)
        
        self.TablePopulation()
        self._table.setColumnWidth(0, 40)   
        self._table.setColumnWidth(1, 55)   
        self._table.setColumnWidth(2, 90)   
        self._table.setColumnWidth(3, 70)   
        self._table.setColumnWidth(4, 145)  

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        # connect preview click -> selection
        self._preview_label.click_callback = self._preview_clicked
        # configure table selection behaviour
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        # Block automatic selection when clicking on cells - only allow programmatic selection
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)

        # Compose side-by-side layout
        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Confirm which detections should be added to the retraining dataset. "
            "Unchecked boxes will be ignored."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        main_h = QHBoxLayout()

        left_v = QVBoxLayout()
        left_v.addWidget(self._preview_label)

        right_v = QVBoxLayout()
        # Keep headers and table on the right side
        right_v.addWidget(self._table)
        right_v.addWidget(buttons)

        main_h.addLayout(left_v)
        main_h.addLayout(right_v)

        # Give the table more room by reducing preview stretch
        main_h.setStretch(0, 2)
        main_h.setStretch(1, 3)

        layout.addLayout(main_h)

        self.updatePreview()

    def TablePopulation(self) -> None:
        self._table.setRowCount(len(self._detections))
        for row, det in enumerate(self._detections):
            # No. column (row index + 1)
            no_item = QTableWidgetItem(str(row + 1))
            no_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 0, no_item)
            
            # Keep checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(det["accepted"])
            checkbox.stateChanged.connect(self._make_keep_handler(row))
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(checkbox)
            self._table.setCellWidget(row, 1, container)

            # Class combo
            combo = QComboBox()
            for value, label in self._class_choices:
                combo.addItem(label, value)
            current_index = combo.findData(det["class_id"])
            if current_index == -1:
                current_index = 0
            combo.setCurrentIndex(current_index)
            combo.currentIndexChanged.connect(self.classHandler(row, combo))
            self._table.setCellWidget(row, 2, combo)

            # Score (read only)
            score_item = QTableWidgetItem(f"{det['score'] * 100:.2f}%")
            score_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 3, score_item)

            # BBox coordinates
            bbox = det["bbox"]
            bbox_text = ", ".join(f"{int(coord)}" for coord in bbox)
            bbox_item = QTableWidgetItem(bbox_text)
            bbox_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 4, bbox_item)

        # make sure no row is selected initially
        self._selected_index = None
        self._table.clearSelection()

    def _make_keep_handler(self, index: int):
        def handler(state: int) -> None:
            self._detections[index]["accepted"] = state == Qt.Checked
            current_selection = self._selected_index
            self.updatePreview()
            # Restore selection after update
            if current_selection is not None:
                self._table.selectRow(current_selection)

        return handler

    def _on_table_selection_changed(self) -> None:
        # update selected index from table selection
        try:
            selected = self._table.selectedIndexes()
            if not selected:
                self._selected_index = None
            else:
                row = selected[0].row()
                self._selected_index = row
        except Exception:
            pass

    def _preview_clicked(self, pos) -> None:
        """Handle clicks on the preview: map the click to original image coords and select matching bbox row."""
        try:
            pix = self._preview_label.pixmap()
            if pix is None:
                return
            pw = pix.width()
            ph = pix.height()
            lw = self._preview_label.width()
            lh = self._preview_label.height()

            # compute offsets if pixmap is letterboxed inside the label
            x_offset = max(0, (lw - pw) // 2)
            y_offset = max(0, (lh - ph) // 2)

            x = pos.x() - x_offset
            y = pos.y() - y_offset
            if x < 0 or y < 0 or x >= pw or y >= ph:
                return

            # map to original image coordinates
            img_w = self._source_image.width()
            img_h = self._source_image.height()
            scale_x = img_w / pw
            scale_y = img_h / ph
            orig_x = x * scale_x
            orig_y = y * scale_y

            # find bbox containing point or label area above bbox (prefer top-most / first match)
            found = None
            for idx, det in enumerate(self._detections):
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                
                # Expand clickable area to include label region above bbox
                # Labels are drawn at (x1, y1-4) with approximate height of 20 pixels
                label_height = 20
                label_y_start = y1 - label_height
                
                # Check if click is within bbox OR in the label area above it
                if x1 <= orig_x <= x2 and label_y_start <= orig_y <= y2:
                    found = idx
                    break

            if found is not None:
                self._selected_index = found
                # select the corresponding row in the table
                self._table.selectRow(found)
                self._table.setCurrentCell(found, 1)  # Set focus to Keep column (column 1, after No.)
        except Exception:
            pass

    def classHandler(self, index: int, combo: QComboBox):
        def handler(_: int) -> None:
            self._detections[index]["class_id"] = combo.currentData()
            # Store current selection before update
            current_selection = self._selected_index
            self.updatePreview()
            # Restore selection after update
            if current_selection is not None:
                self._table.selectRow(current_selection)

        return handler

    def updatePreview(self) -> None:
        pixmap = QPixmap.fromImage(self._source_image)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        for idx, det in enumerate(self._detections):
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
            text = f"{idx + 1}. {class_label} ({score * 100:.1f}%)"
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
