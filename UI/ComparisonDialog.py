# ComparisonDialog.py
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ComparisonDialog(QDialog):
    def __init__(self, parent, champion_scores, challenger_scores):
        super().__init__(parent)
        self.setWindowTitle("Retraining Summary & Deployment")
        self.setMinimumWidth(500)

        self.champion_scores = champion_scores or {} # Handle None
        self.challenger_scores = challenger_scores or {}

        layout = QVBoxLayout(self)

        # Main prompt label
        main_label = QLabel("Retraining and evaluation complete.")
        main_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(main_label)

        # Comparison layout
        comparison_layout = QHBoxLayout()

        # Champion (Current Model) Group
        champion_group = QGroupBox("Current Model (Champion)")
        champion_layout = QVBoxLayout()
        self.populate_metrics_layout(champion_layout, self.champion_scores)
        champion_group.setLayout(champion_layout)
        comparison_layout.addWidget(champion_group)

        # Challenger (New Model) Group
        challenger_group = QGroupBox("New Model (Challenger)")
        challenger_layout = QVBoxLayout()
        self.populate_metrics_layout(challenger_layout, self.challenger_scores, is_challenger=True)
        challenger_group.setLayout(challenger_layout)
        comparison_layout.addWidget(challenger_group)

        layout.addLayout(comparison_layout)
        
        # Deployment prompt
        deploy_label = QLabel("The new model shows the performance above. Do you want to deploy it?")
        deploy_label.setWordWrap(True)
        layout.addWidget(deploy_label)

        # Button Box
        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def populate_metrics_layout(self, layout, scores, is_challenger=False):
        # Safely get scores with a default value
        ap = scores.get("bbox", {}).get("AP", 0.0)
        ap50 = scores.get("bbox", {}).get("AP50", 0.0)
        ap75 = scores.get("bbox", {}).get("AP75", 0.0)

        ap_label = QLabel(f"<b>Overall AP:</b> {ap:.2f}")
        ap50_label = QLabel(f"AP50 (IoU > 0.5): {ap50:.2f}")
        ap75_label = QLabel(f"AP75 (IoU > 0.75): {ap75:.2f}")
        
        layout.addWidget(ap_label)
        layout.addWidget(ap50_label)
        layout.addWidget(ap75_label)

        # If it's the challenger, add a visual delta
        if is_challenger:
            champion_ap = self.champion_scores.get("bbox", {}).get("AP", 0.0)
            delta = ap - champion_ap
            
            delta_label = QLabel(f"Change: {delta:+.2f} AP")
            if delta > 0:
                delta_label.setStyleSheet("color: green; font-weight: bold;")
            elif delta < 0:
                delta_label.setStyleSheet("color: red; font-weight: bold;")
            
            layout.addSpacing(10)
            layout.addWidget(delta_label)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    square_app = ComparisonDialog(None, {}, {})
    square_app.show()
    sys.exit(app.exec_())