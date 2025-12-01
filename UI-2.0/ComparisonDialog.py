# ComparisonDialog.py
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ComparisonDialog(QDialog):
    def __init__(self, parent, champion_scores, challenger_scores):
        super().__init__(parent)
        self.setWindowTitle("Retraining Summary & Deployment")
        self.setMinimumWidth(600)

        self.champion_scores = champion_scores or {} # Handle None
        self.challenger_scores = challenger_scores or {}
        self.advanced_visible = False

        layout = QVBoxLayout(self)

        # Main prompt label
        main_label = QLabel("Retraining Complete!")
        main_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #4CAF50;")
        main_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(main_label)
        
        layout.addSpacing(10)

        # Simple summary section
        summary_group = QGroupBox("Performance Summary")
        summary_layout = QVBoxLayout()
        self.populate_simple_summary(summary_layout)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Advanced metrics section (collapsible)
        self.advanced_group = QGroupBox("Advanced Metrics")
        self.advanced_layout = QVBoxLayout()
        self.populate_advanced_metrics()
        self.advanced_group.setLayout(self.advanced_layout)
        self.advanced_group.setVisible(False)
        layout.addWidget(self.advanced_group)
        
        # Toggle button for advanced metrics
        self.toggle_button = QPushButton("Show Advanced Metrics")
        self.toggle_button.setStyleSheet("text-align: left; padding: 8px; font-weight: bold;")
        self.toggle_button.clicked.connect(self.toggle_advanced_metrics)
        layout.addWidget(self.toggle_button)
        
        layout.addSpacing(10)
        
        # Deployment prompt
        deploy_label = QLabel("Do you want to deploy the new model and use it for detection?")
        deploy_label.setWordWrap(True)
        deploy_label.setStyleSheet("font-size: 11pt;")
        layout.addWidget(deploy_label)
        
        help_label = QLabel("(Click 'Yes' to replace your current model, or 'No' to keep the existing one)")
        help_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(help_label)

        # Button Box
        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        button_box.button(QDialogButtonBox.Yes).setStyleSheet("padding: 8px 20px; font-size: 11pt;")
        button_box.button(QDialogButtonBox.No).setStyleSheet("padding: 8px 20px; font-size: 11pt;")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def populate_simple_summary(self, layout):
        """Show easy-to-understand metrics for non-technical users"""
        # Get AP scores
        champion_ap = self.champion_scores.get("bbox", {}).get("AP", 0.0)
        challenger_ap = self.challenger_scores.get("bbox", {}).get("AP", 0.0)
        
        # Calculate improvement
        improvement = challenger_ap - champion_ap
        improvement_percent = (improvement / champion_ap * 100) if champion_ap > 0 else 0
        
        # Convert AP to detection accuracy percentage (AP is 0-100 scale)
        old_accuracy = champion_ap
        new_accuracy = challenger_ap
        
        # Overall verdict based on both metrics
        verdict_label = QLabel()
        verdict_label.setWordWrap(True)
        verdict_label.setStyleSheet("font-size: 12pt; padding: 10px; border-radius: 5px;")
        
        if improvement > 2.0:  # Significant improvement
            verdict_label.setText(
                f"<b>Great improvement!</b> The new model performs <b>{improvement:.1f} points better</b> "
                f"({improvement_percent:+.1f}%) than your current model."
            )
            verdict_label.setStyleSheet(verdict_label.styleSheet() + "background-color: #E8F5E9; color: #2E7D32;")
        elif improvement > 0.5:  # Modest improvement
            verdict_label.setText(
                f"<b>Improvement detected.</b> The new model is <b>{improvement:.1f} points better</b> "
                f"({improvement_percent:+.1f}%). Consider deploying it."
            )
            verdict_label.setStyleSheet(verdict_label.styleSheet() + "background-color: #FFF9C4; color: #F57F17;")
        elif improvement > -0.5:  # Marginal change
            verdict_label.setText(
                f"<b>Similar performance.</b> The new model is about the same as your current model "
                f"({improvement:+.1f} points difference)."
            )
            verdict_label.setStyleSheet(verdict_label.styleSheet() + "background-color: #E3F2FD; color: #1565C0;")
        else:  # Performance declined
            verdict_label.setText(
                f"<b>Performance decreased.</b> The new model is <b>{abs(improvement):.1f} points worse</b> "
                f"({improvement_percent:.1f}%) than your current model. Deployment not recommended."
            )
            verdict_label.setStyleSheet(verdict_label.styleSheet() + "background-color: #FFEBEE; color: #C62828;")
        
        layout.addWidget(verdict_label)
        
        layout.addSpacing(15)
        
        # Simple comparison table
        comparison_label = QLabel("<b>Detection Accuracy Comparison:</b>")
        comparison_label.setStyleSheet("font-size: 11pt;")
        layout.addWidget(comparison_label)
        
        table = QTableWidget(2, 2)
        table.setHorizontalHeaderLabels(["Current Model", "New Model"])
        table.verticalHeader().setVisible(False)
        table.setMaximumHeight(100)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Row 1: Accuracy scores
        old_item = QTableWidgetItem(f"{old_accuracy:.1f}%")
        old_item.setTextAlignment(Qt.AlignCenter)
        old_item.setFont(QFont("Arial", 14, QFont.Bold))
        table.setItem(0, 0, old_item)
        
        new_item = QTableWidgetItem(f"{new_accuracy:.1f}%")
        new_item.setTextAlignment(Qt.AlignCenter)
        new_item.setFont(QFont("Arial", 14, QFont.Bold))
        if improvement > 0:
            new_item.setForeground(QColor("#4CAF50"))
        elif improvement < 0:
            new_item.setForeground(QColor("#F44336"))
        table.setItem(0, 1, new_item)
        
        # Row 2: Change indicator
        change_item = QTableWidgetItem("—")
        change_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(1, 0, change_item)
        
        change_text = f"{improvement:+.1f}% ({improvement_percent:+.1f}%)"
        change_indicator = QTableWidgetItem(change_text)
        change_indicator.setTextAlignment(Qt.AlignCenter)
        change_indicator.setFont(QFont("Arial", 10, QFont.Bold))
        if improvement > 0:
            change_indicator.setForeground(QColor("#4CAF50"))
        elif improvement < 0:
            change_indicator.setForeground(QColor("#F44336"))
        table.setItem(1, 1, change_indicator)
        
        layout.addWidget(table)
        
        # Explanation
        explain_label = QLabel(
            "<i>Higher accuracy means the model is better at detecting microplastics correctly.</i>"
        )
        explain_label.setStyleSheet("color: #666; font-size: 9pt;")
        explain_label.setWordWrap(True)
        layout.addWidget(explain_label)

    def populate_advanced_metrics(self):
        """Show detailed AP metrics for technical users"""
        # Comparison layout
        comparison_layout = QHBoxLayout()

        # Champion (Current Model) Group
        champion_group = QGroupBox("Current Model Details")
        champion_layout = QVBoxLayout()
        self.populate_technical_metrics(champion_layout, self.champion_scores)
        champion_group.setLayout(champion_layout)
        comparison_layout.addWidget(champion_group)

        # Challenger (New Model) Group
        challenger_group = QGroupBox("New Model Details")
        challenger_layout = QVBoxLayout()
        self.populate_technical_metrics(challenger_layout, self.challenger_scores)
        challenger_group.setLayout(challenger_layout)
        comparison_layout.addWidget(challenger_group)

        # Change/Delta Group (separate column)
        change_group = QGroupBox("Change")
        change_layout = QVBoxLayout()
        self.populate_change_metrics(change_layout)
        change_group.setLayout(change_layout)
        comparison_layout.addWidget(change_group)

        self.advanced_layout.addLayout(comparison_layout)

    def populate_technical_metrics(self, layout, scores):
        """Populate detailed AP metrics"""
        # Safely get scores with a default value
        ap = scores.get("bbox", {}).get("AP", 0.0)
        ap50 = scores.get("bbox", {}).get("AP50", 0.0)
        ap75 = scores.get("bbox", {}).get("AP75", 0.0)

        ap_label = QLabel(f"<b>Overall AP:</b> {ap:.2f}")
        ap50_label = QLabel(f"AP50 (IoU ≥ 0.5): {ap50:.2f}")
        ap75_label = QLabel(f"AP75 (IoU ≥ 0.75): {ap75:.2f}")
        
        layout.addWidget(ap_label)
        layout.addWidget(ap50_label)
        layout.addWidget(ap75_label)

    def populate_change_metrics(self, layout):
        """Populate the change/delta metrics in separate column"""
        champion_ap = self.champion_scores.get("bbox", {}).get("AP", 0.0)
        champion_ap50 = self.champion_scores.get("bbox", {}).get("AP50", 0.0)
        champion_ap75 = self.champion_scores.get("bbox", {}).get("AP75", 0.0)
        
        challenger_ap = self.challenger_scores.get("bbox", {}).get("AP", 0.0)
        challenger_ap50 = self.challenger_scores.get("bbox", {}).get("AP50", 0.0)
        challenger_ap75 = self.challenger_scores.get("bbox", {}).get("AP75", 0.0)
        
        delta_ap = challenger_ap - champion_ap
        delta_ap50 = challenger_ap50 - champion_ap50
        delta_ap75 = challenger_ap75 - champion_ap75
        
        # Overall AP change
        ap_delta_label = QLabel(f"<b>{delta_ap:+.2f}</b>")
        if delta_ap > 0:
            ap_delta_label.setStyleSheet("color: green; font-weight: bold;")
        elif delta_ap < 0:
            ap_delta_label.setStyleSheet("color: red; font-weight: bold;")
        
        # AP50 change
        ap50_delta_label = QLabel(f"{delta_ap50:+.2f}")
        if delta_ap50 > 0:
            ap50_delta_label.setStyleSheet("color: green; font-weight: bold;")
        elif delta_ap50 < 0:
            ap50_delta_label.setStyleSheet("color: red; font-weight: bold;")
        
        # AP75 change
        ap75_delta_label = QLabel(f"{delta_ap75:+.2f}")
        if delta_ap75 > 0:
            ap75_delta_label.setStyleSheet("color: green; font-weight: bold;")
        elif delta_ap75 < 0:
            ap75_delta_label.setStyleSheet("color: red; font-weight: bold;")
        
        layout.addWidget(ap_delta_label)
        layout.addWidget(ap50_delta_label)
        layout.addWidget(ap75_delta_label)
    
    def toggle_advanced_metrics(self):
        """Toggle visibility of advanced metrics section"""
        self.advanced_visible = not self.advanced_visible
        self.advanced_group.setVisible(self.advanced_visible)
        
        if self.advanced_visible:
            self.toggle_button.setText("Hide Advanced Metrics")
        else:
            self.toggle_button.setText("Show Advanced Metrics")
        
        # Adjust window size after toggling
        self.adjustSize()
        self.setMinimumHeight(0)  # Reset minimum height to allow shrinking
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Example data: Improved model scenario
    champion_scores = {
        "bbox": {
            "AP": 65.3,
            "AP50": 82.1,
            "AP75": 70.5
        }
    }
    
    challenger_scores = {
        "bbox": {
            "AP": 72.8,
            "AP50": 89.3,
            "AP75": 78.2
        }
    }
    
    # Example data: Worse model scenario
    # champion_scores = {
    #     "bbox": {
    #         "AP": 72.5,
    #         "AP50": 85.3,
    #         "AP75": 76.8
    #     }
    # }
    
    # challenger_scores = {
    #     "bbox": {
    #         "AP": 68.2,
    #         "AP50": 79.1,
    #         "AP75": 71.4
    #     }
    # }
    
    dialog = ComparisonDialog(None, champion_scores, challenger_scores)
    result = dialog.exec_()
    
    if result == QDialog.Accepted:
        print("User chose to deploy the new model")
    else:
        print("User chose to keep the current model")
    
    sys.exit()