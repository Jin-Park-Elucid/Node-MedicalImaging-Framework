"""
PyQt5-based GUI workflow editor for medical imaging framework.

Provides visual node graph editing, execution, and monitoring.
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QDockWidget, QListWidget, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QBrush, QColor

from ..core import NodeRegistry, ComputationalGraph, GraphExecutor
from .node_graphics import NodeGraphicsItem, ConnectionGraphicsItem


class NodeGraphEditor(QMainWindow):
    """
    Main GUI editor for creating and executing node-based workflows.

    Features:
    - Visual node graph editing
    - Node library browser
    - Properties panel
    - Graph execution
    - Save/load workflows
    """

    def __init__(self):
        super().__init__()
        self.graph = ComputationalGraph("New Workflow")
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Set editor reference on scene so nodes can access the graph
        self.scene.editor = self

        # Track graphics items
        self.node_graphics = {}  # node_name -> NodeGraphicsItem
        self.connection_graphics = []  # List of ConnectionGraphicsItem

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Medical Imaging Framework - Workflow Editor")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget - node graph view
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)  # Pan with mouse
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # Enable mouse wheel zoom
        self.view.wheelEvent = self.wheel_event_handler

        self.setCentralWidget(self.view)

        # Set dark background
        self.scene.setBackgroundBrush(QBrush(QColor(50, 50, 50)))
        self.scene.setSceneRect(-2000, -2000, 4000, 4000)  # Large scene

        # Create docks
        self.create_node_palette()
        self.create_control_panel()

        # Create menu bar
        self.create_menu_bar()

        # Status bar
        self.statusBar().showMessage("Ready - Load a workflow or create nodes from the library")

    def create_node_palette(self):
        """Create the node library palette."""
        dock = QDockWidget("Node Library", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Available Nodes:")
        title.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        # Node list organized by category
        categories = NodeRegistry.get_categories()
        for category in sorted(categories):
            category_label = QLabel(f"\n{category.upper()}:")
            category_label.setStyleSheet("color: #4A90E2; font-weight: bold;")
            layout.addWidget(category_label)

            nodes = NodeRegistry.get_nodes_by_category(category)
            for node_name in sorted(nodes):
                info = NodeRegistry.get_node_info(node_name)
                btn = QPushButton(f"{node_name}")
                btn.setToolTip(info['description'])
                btn.clicked.connect(lambda checked, n=node_name: self.add_node(n))
                layout.addWidget(btn)

        layout.addStretch()

        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = file_menu.addAction("&New Workflow")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.clear_workflow)

        load_action = file_menu.addAction("&Load Workflow")
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_workflow)

        save_action = file_menu.addAction("&Save Workflow")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_workflow)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_in_action = view_menu.addAction("Zoom &In")
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(lambda: self.view.scale(1.2, 1.2))

        zoom_out_action = view_menu.addAction("Zoom &Out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(lambda: self.view.scale(0.8, 0.8))

        fit_action = view_menu.addAction("&Fit to View")
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_to_view)

        view_menu.addSeparator()

        reset_view_action = view_menu.addAction("&Reset View")
        reset_view_action.triggered.connect(self.reset_view)

        # Workflow menu
        workflow_menu = menubar.addMenu("&Workflow")

        validate_action = workflow_menu.addAction("&Validate")
        validate_action.setShortcut("Ctrl+V")
        validate_action.triggered.connect(self.validate_workflow)

        execute_action = workflow_menu.addAction("&Execute")
        execute_action.setShortcut("Ctrl+E")
        execute_action.triggered.connect(self.execute_workflow)

        workflow_menu.addSeparator()

        auto_layout_action = workflow_menu.addAction("Auto-&Layout Nodes")
        auto_layout_action.setShortcut("Ctrl+L")
        auto_layout_action.triggered.connect(self.auto_layout)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = help_menu.addAction("&About")
        about_action.triggered.connect(self.show_about)

    def fit_to_view(self):
        """Fit the graph to the view."""
        if self.scene.items():
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.view.scale(0.9, 0.9)

    def reset_view(self):
        """Reset view transformation."""
        self.view.resetTransform()

    def auto_layout(self):
        """Auto-layout all nodes."""
        self._auto_layout_nodes()
        self.draw_graph()

    def show_about(self):
        """Show about dialog."""
        msg = "Medical Imaging Framework\n"
        msg += "Node-Based Workflow Editor\n\n"
        msg += "Version: 1.0\n\n"
        msg += "Features:\n"
        msg += "• Visual node graph editing\n"
        msg += "• Drag and drop nodes\n"
        msg += "• Zoom and pan\n"
        msg += "• Save/load workflows\n"
        msg += "• Execute pipelines\n\n"
        msg += "Controls:\n"
        msg += "• Left click + drag: Pan\n"
        msg += "• Mouse wheel: Zoom\n"
        msg += "• Right click node: Context menu\n"

        QMessageBox.about(self, "About", msg)

    def create_control_panel(self):
        """Create the control panel."""
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout()

        # Workflow info
        info_label = QLabel("Workflow Information")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(info_label)

        self.workflow_info = QLabel("Nodes: 0\nConnections: 0")
        self.workflow_info.setStyleSheet("padding: 10px; background: #2A2A2A; border-radius: 5px;")
        layout.addWidget(self.workflow_info)

        # Buttons
        layout.addSpacing(20)

        execute_btn = QPushButton("▶ Execute Workflow")
        execute_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        execute_btn.clicked.connect(self.execute_workflow)
        layout.addWidget(execute_btn)

        validate_btn = QPushButton("✓ Validate Workflow")
        validate_btn.clicked.connect(self.validate_workflow)
        layout.addWidget(validate_btn)

        layout.addSpacing(20)

        save_btn = QPushButton("💾 Save Workflow")
        save_btn.clicked.connect(self.save_workflow)
        layout.addWidget(save_btn)

        load_btn = QPushButton("📂 Load Workflow")
        load_btn.clicked.connect(self.load_workflow)
        layout.addWidget(load_btn)

        clear_btn = QPushButton("🗑 Clear All")
        clear_btn.clicked.connect(self.clear_workflow)
        layout.addWidget(clear_btn)

        layout.addStretch()

        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def add_node(self, node_class_name: str):
        """Add a node to the graph."""
        node_count = len(self.graph.nodes)
        node_name = f"{node_class_name}_{node_count}"

        node = NodeRegistry.create_node(node_class_name, node_name)
        if node:
            # Set position in center of visible area
            view_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
            node.position = (view_rect.center().x() - 90, view_rect.center().y() - 50)

            self.graph.add_node(node)

            # Draw the node
            self._draw_node(node_name, node)

            self.update_workflow_info()
            self.statusBar().showMessage(f"Added node: {node_name}")

            # Zoom to show the node if it's the first one
            if len(self.graph.nodes) == 1:
                self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
                self.view.scale(0.9, 0.9)

    def execute_workflow(self):
        """Execute the current workflow."""
        if len(self.graph.nodes) == 0:
            QMessageBox.warning(self, "No Nodes", "Add nodes to the workflow first!")
            return

        # Validate first
        is_valid, errors = self.graph.validate()
        if not is_valid:
            QMessageBox.critical(
                self,
                "Validation Failed",
                "Workflow has errors:\n\n" + "\n".join(errors)
            )
            return

        try:
            self.statusBar().showMessage("Executing workflow...")
            executor = GraphExecutor(self.graph)
            result = executor.execute()

            if result.is_successful():
                QMessageBox.information(
                    self,
                    "Success",
                    f"Workflow executed successfully!\n\n"
                    f"Time: {result.execution_time:.2f}s\n"
                    f"Nodes executed: {len(result.node_results)}"
                )
            else:
                error_msg = "\n".join([f"{k}: {v}" for k, v in result.get_errors().items()])
                QMessageBox.warning(
                    self,
                    "Execution Failed",
                    f"Workflow execution failed:\n\n{error_msg}"
                )

            self.statusBar().showMessage(f"Execution completed: {result.status.value}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Execution error:\n\n{str(e)}")

    def validate_workflow(self):
        """Validate the current workflow."""
        is_valid, errors = self.graph.validate()

        if is_valid:
            QMessageBox.information(
                self,
                "Validation Successful",
                "Workflow is valid and ready to execute!"
            )
        else:
            QMessageBox.warning(
                self,
                "Validation Failed",
                "Workflow has errors:\n\n" + "\n".join(errors)
            )

    def save_workflow(self):
        """Save workflow to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow",
            "",
            "JSON Files (*.json)"
        )

        if filename:
            try:
                # Update node positions from graphics items
                for node_name, node_gfx in self.node_graphics.items():
                    node = self.graph.nodes[node_name]
                    node.position = (node_gfx.pos().x(), node_gfx.pos().y())

                self.graph.save_to_file(filename)
                self.statusBar().showMessage(f"Workflow saved: {filename}")
                QMessageBox.information(self, "Saved",
                    f"Workflow saved to:\n{filename}\n\n"
                    f"Node positions have been preserved.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def load_workflow(self):
        """Load workflow from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Workflow",
            "",
            "JSON Files (*.json)"
        )

        if filename:
            try:
                self.graph.load_from_file(filename)

                # Draw the graph visually
                self.draw_graph()

                self.update_workflow_info()
                self.statusBar().showMessage(f"Workflow loaded: {filename}")

                # Show detailed information about loaded workflow
                node_list = "\n".join([f"  • {name} ({node.__class__.__name__.replace('Node', '')})"
                                      for name, node in list(self.graph.nodes.items())[:15]])
                if len(self.graph.nodes) > 15:
                    node_list += f"\n  ... and {len(self.graph.nodes) - 15} more nodes"

                msg = f"✓ Workflow loaded and displayed!\n\n"
                msg += f"File: {Path(filename).name}\n"
                msg += f"Nodes: {len(self.graph.nodes)}\n"
                msg += f"Connections: {len(self.graph.links)}\n\n"

                if len(self.graph.nodes) > 0:
                    msg += f"Loaded Nodes:\n{node_list}\n\n"

                msg += "Visual Features:\n"
                msg += "• Drag nodes to reposition\n"
                msg += "• Click nodes to select\n"
                msg += "• Right-click for context menu\n"
                msg += "• Mouse wheel to zoom\n\n"

                msg += "To execute this workflow, use the Python script:\n"
                msg += "python examples/medical_segmentation_pipeline/train_pipeline.py"

                QMessageBox.information(self, "Workflow Loaded", msg)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                QMessageBox.critical(self, "Error", f"Failed to load workflow:\n\n{str(e)}\n\nDetails:\n{error_details[:500]}")

    def clear_workflow(self):
        """Clear the current workflow."""
        reply = QMessageBox.question(
            self,
            "Clear Workflow",
            "Clear all nodes and connections?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.graph = ComputationalGraph("New Workflow")
            self.clear_canvas()
            self.update_workflow_info()
            self.statusBar().showMessage("Workflow cleared")

    def update_workflow_info(self):
        """Update the workflow information display."""
        info_text = f"Workflow: {self.graph.name}\n"
        info_text += f"Nodes: {len(self.graph.nodes)}\n"
        info_text += f"Connections: {len(self.graph.links)}\n"

        if len(self.graph.nodes) > 0:
            info_text += "\nLoaded Nodes:\n"
            for name, node in list(self.graph.nodes.items())[:10]:  # Show first 10
                node_type = node.__class__.__name__.replace('Node', '')
                info_text += f"  • {name} ({node_type})\n"

            if len(self.graph.nodes) > 10:
                info_text += f"  ... and {len(self.graph.nodes) - 10} more\n"

        self.workflow_info.setText(info_text)

    def clear_canvas(self):
        """Clear all graphics items from the canvas."""
        # Remove all items
        self.scene.clear()

        # Clear tracking dictionaries
        self.node_graphics.clear()
        self.connection_graphics.clear()

    def draw_graph(self):
        """Draw the entire graph on the canvas."""
        # Clear existing graphics
        self.clear_canvas()

        # Auto-layout nodes if needed
        self._auto_layout_nodes()

        # Draw nodes
        for node_name, node in self.graph.nodes.items():
            self._draw_node(node_name, node)

        # Draw connections
        for link in self.graph.links:
            self._draw_connection(link)

        # Fit view to content
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.view.scale(0.9, 0.9)  # Zoom out slightly for margins

    def _auto_layout_nodes(self):
        """Automatically layout nodes if they don't have positions."""
        nodes_without_pos = [node for node in self.graph.nodes.values()
                            if not hasattr(node, 'position') or node.position == (0, 0)]

        if nodes_without_pos:
            # Simple grid layout
            cols = max(3, int(len(self.graph.nodes) ** 0.5))
            spacing_x = 250
            spacing_y = 150

            for i, node in enumerate(self.graph.nodes.values()):
                if not hasattr(node, 'position') or node.position == (0, 0):
                    row = i // cols
                    col = i % cols
                    node.position = (col * spacing_x + 100, row * spacing_y + 100)

    def _draw_node(self, node_name, node):
        """Draw a single node on the canvas."""
        # Determine node category
        category = self._get_node_category(node)

        # Create graphics item
        node_item = NodeGraphicsItem(node, category)

        # Add to scene
        self.scene.addItem(node_item)

        # Track it
        self.node_graphics[node_name] = node_item

    def _get_node_category(self, node):
        """Get the category of a node from the registry."""
        node_class_name = node.__class__.__name__

        # Search registry for this node
        for reg_name, info in NodeRegistry._registry.items():
            if info['class'].__name__ == node_class_name:
                return info['category']

        return 'data'  # Default category

    def _draw_connection(self, link):
        """Draw a connection between two nodes."""
        # Link has source and target which are Port objects
        # Port objects have .node attribute
        source_node_name = link.source.node.name
        target_node_name = link.target.node.name

        # Get graphics items
        source_node_gfx = self.node_graphics.get(source_node_name)
        target_node_gfx = self.node_graphics.get(target_node_name)

        if not source_node_gfx or not target_node_gfx:
            return

        # Find the port graphics items
        source_port_gfx = None
        target_port_gfx = None

        # Find output port on source (link.source is a Port, .name is port name)
        for port_gfx in source_node_gfx.output_ports:
            if port_gfx.port_name == link.source.name:
                source_port_gfx = port_gfx
                break

        # Find input port on target
        for port_gfx in target_node_gfx.input_ports:
            if port_gfx.port_name == link.target.name:
                target_port_gfx = port_gfx
                break

        if source_port_gfx and target_port_gfx:
            # Create connection
            connection = ConnectionGraphicsItem(source_port_gfx, target_port_gfx)
            self.scene.addItem(connection)
            self.connection_graphics.append(connection)

    def wheel_event_handler(self, event):
        """Handle mouse wheel for zooming."""
        # Zoom factor
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Set anchors
        self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.view.setResizeAnchor(QGraphicsView.NoAnchor)

        # Save the scene position
        old_pos = self.view.mapToScene(event.pos())

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.view.scale(zoom_factor, zoom_factor)

        # Get the new position
        new_pos = self.view.mapToScene(event.pos())

        # Move scene to old position
        delta = new_pos - old_pos
        self.view.translate(delta.x(), delta.y())


def main():
    """Launch the GUI editor."""
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(palette)

    # Import nodes to trigger registration
    import medical_imaging_framework.nodes

    editor = NodeGraphEditor()
    editor.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
