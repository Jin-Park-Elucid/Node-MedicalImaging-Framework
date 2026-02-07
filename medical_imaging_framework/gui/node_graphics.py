"""
Graphics items for visual node representation in the GUI.

Provides visual representation of nodes and connections on the canvas.
"""

from PyQt5.QtWidgets import (
    QGraphicsItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QMenu, QAction, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QFormLayout, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt5.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath, QLinearGradient
)
import math


class NodeGraphicsItem(QGraphicsItem):
    """
    Visual representation of a node on the canvas.

    Features:
    - Rounded rectangle with gradient
    - Title bar with node name
    - Input/output ports
    - Color-coded by category
    - Interactive (selectable, movable)
    """

    # Category colors
    CATEGORY_COLORS = {
        'data': QColor(70, 130, 180),      # Steel Blue
        'networks': QColor(34, 139, 34),    # Forest Green
        'training': QColor(220, 20, 60),    # Crimson
        'inference': QColor(255, 140, 0),   # Dark Orange
        'visualization': QColor(138, 43, 226)  # Blue Violet
    }

    def __init__(self, node, category='data'):
        super().__init__()
        self.node = node
        self.category = category

        # Dimensions
        self.width = 180
        self.height = 100
        self.title_height = 30
        self.port_radius = 8  # Increased for easier clicking

        # Visual properties
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

        # State
        self.hovered = False

        # Create child items
        self._create_title()
        self._create_body_text()
        self._create_ports()

        # Update position from node
        if hasattr(node, 'position') and node.position:
            self.setPos(node.position[0], node.position[1])

    def _create_title(self):
        """Create the title text."""
        self.title_item = QGraphicsTextItem(self.node.name, self)
        self.title_item.setDefaultTextColor(Qt.white)
        font = QFont("Arial", 10, QFont.Bold)
        self.title_item.setFont(font)

        # Center the title
        title_rect = self.title_item.boundingRect()
        self.title_item.setPos(
            (self.width - title_rect.width()) / 2,
            5
        )

    def _create_body_text(self):
        """Create body text showing node type."""
        node_type = self.node.__class__.__name__.replace('Node', '')
        self.body_item = QGraphicsTextItem(node_type, self)
        self.body_item.setDefaultTextColor(QColor(200, 200, 200))
        font = QFont("Arial", 8)
        self.body_item.setFont(font)

        # Center the body text
        body_rect = self.body_item.boundingRect()
        self.body_item.setPos(
            (self.width - body_rect.width()) / 2,
            self.title_height + 10
        )

        # Add port counts
        num_inputs = len(self.node.inputs)
        num_outputs = len(self.node.outputs)
        port_text = f"In: {num_inputs} | Out: {num_outputs}"

        self.ports_item = QGraphicsTextItem(port_text, self)
        self.ports_item.setDefaultTextColor(QColor(150, 150, 150))
        font = QFont("Arial", 7)
        self.ports_item.setFont(font)

        ports_rect = self.ports_item.boundingRect()
        self.ports_item.setPos(
            (self.width - ports_rect.width()) / 2,
            self.title_height + 30
        )

    def _create_ports(self):
        """Create visual port indicators."""
        self.input_ports = []
        self.output_ports = []

        # Input ports (left side)
        num_inputs = len(self.node.inputs)
        if num_inputs > 0:
            spacing = (self.height - self.title_height) / (num_inputs + 1)
            for i, (port_name, port) in enumerate(self.node.inputs.items()):
                y = self.title_height + spacing * (i + 1)
                port_item = PortGraphicsItem(port, 'input', port_name, self)
                port_item.setPos(-self.port_radius, y)
                self.input_ports.append(port_item)

        # Output ports (right side)
        num_outputs = len(self.node.outputs)
        if num_outputs > 0:
            spacing = (self.height - self.title_height) / (num_outputs + 1)
            for i, (port_name, port) in enumerate(self.node.outputs.items()):
                y = self.title_height + spacing * (i + 1)
                port_item = PortGraphicsItem(port, 'output', port_name, self)
                port_item.setPos(self.width - self.port_radius, y)
                self.output_ports.append(port_item)

    def boundingRect(self):
        """Return the bounding rectangle."""
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget):
        """Paint the node."""
        painter.setRenderHint(QPainter.Antialiasing)

        # Get base color for category
        base_color = self.CATEGORY_COLORS.get(self.category, QColor(100, 100, 100))

        # Adjust color based on selection/hover
        if self.isSelected():
            border_color = QColor(255, 255, 100)
            border_width = 3
        elif self.hovered:
            border_color = QColor(200, 200, 200)
            border_width = 2
        else:
            border_color = QColor(80, 80, 80)
            border_width = 1

        # Draw shadow
        shadow_offset = 3
        shadow_rect = QRectF(shadow_offset, shadow_offset, self.width, self.height)
        painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(shadow_rect, 8, 8)

        # Draw main body with gradient
        gradient = QLinearGradient(0, 0, 0, self.height)
        gradient.setColorAt(0, base_color.lighter(120))
        gradient.setColorAt(1, base_color.darker(110))

        body_rect = QRectF(0, 0, self.width, self.height)
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(body_rect, 8, 8)

        # Draw title bar separator
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawLine(0, self.title_height, self.width, self.title_height)

        # Draw title bar background
        title_gradient = QLinearGradient(0, 0, 0, self.title_height)
        title_gradient.setColorAt(0, base_color.lighter(130))
        title_gradient.setColorAt(1, base_color)

        title_path = QPainterPath()
        title_path.addRoundedRect(QRectF(0, 0, self.width, self.title_height), 8, 8)
        title_path.addRect(QRectF(0, self.title_height - 8, self.width, 8))
        painter.setBrush(QBrush(title_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawPath(title_path)

    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsItem.ItemPositionHasChanged:
            # Update node position
            self.node.position = (self.pos().x(), self.pos().y())
            # Update connected edges
            for port in self.input_ports + self.output_ports:
                port.update_connections()

        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        """Show context menu."""
        menu = QMenu()

        # Edit Parameters - most common action, put first
        edit_action = QAction("⚙ Edit Parameters", None)
        edit_action.triggered.connect(self.edit_parameters)
        menu.addAction(edit_action)

        menu.addSeparator()

        info_action = QAction("Node Info", None)
        info_action.triggered.connect(self.show_info)
        menu.addAction(info_action)

        config_action = QAction("View Configuration", None)
        config_action.triggered.connect(self.show_config)
        menu.addAction(config_action)

        menu.addSeparator()

        delete_action = QAction("Delete Node", None)
        delete_action.triggered.connect(self.delete_node)
        menu.addAction(delete_action)

        menu.exec_(event.screenPos())

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to edit parameters."""
        if event.button() == Qt.LeftButton:
            self.edit_parameters()
        super().mouseDoubleClickEvent(event)

    def edit_parameters(self):
        """Open parameter editing dialog."""
        dialog = NodeParameterDialog(self.node, parent=None)

        if dialog.exec_() == QDialog.Accepted:
            # Get values from dialog
            new_values = dialog.get_values()

            # Check which values actually changed
            changed_fields = []
            for field_name, new_value in new_values.items():
                old_value = str(self.node.config.get(field_name, ''))
                new_value_str = str(new_value)

                if old_value != new_value_str:
                    changed_fields.append(field_name)
                    self.node.config[field_name] = new_value

            # Only show confirmation if something actually changed
            if changed_fields:
                QMessageBox.information(
                    None,
                    "Parameters Updated",
                    f"Parameters for '{self.node.name}' have been updated.\n\n"
                    f"Updated fields: {', '.join(changed_fields)}"
                )

    def show_info(self):
        """Show node information."""
        info = f"Node: {self.node.name}\n"
        info += f"Type: {self.node.__class__.__name__}\n"
        info += f"Category: {self.category}\n\n"

        info += f"Inputs ({len(self.node.inputs)}):\n"
        for port_name, port in self.node.inputs.items():
            info += f"  • {port_name} ({port.data_type.value})\n"

        info += f"\nOutputs ({len(self.node.outputs)}):\n"
        for port_name, port in self.node.outputs.items():
            info += f"  • {port_name} ({port.data_type.value})\n"

        QMessageBox.information(None, "Node Information", info)

    def show_config(self):
        """Show node configuration."""
        config = f"Configuration for {self.node.name}:\n\n"
        if self.node.config:
            for key, value in self.node.config.items():
                config += f"  {key}: {value}\n"
        else:
            config += "  (No configuration)"

        QMessageBox.information(None, "Node Configuration", config)

    def delete_node(self):
        """Delete this node."""
        # Remove from scene
        if self.scene():
            self.scene().removeItem(self)


class PortGraphicsItem(QGraphicsEllipseItem):
    """Visual representation of a port."""

    def __init__(self, port, port_type, port_name, parent=None):
        # Make the port circle bigger for easier clicking (radius 8 instead of 6)
        super().__init__(-8, -8, 16, 16, parent)
        self.port = port
        self.port_type = port_type  # 'input' or 'output'
        self.port_name = port_name
        self.connections = []

        # Connection dragging state
        self.dragging_connection = None
        self.temp_line = None

        # Visual style
        if port_type == 'input':
            self.setBrush(QBrush(QColor(100, 150, 255)))
        else:
            self.setBrush(QBrush(QColor(255, 150, 100)))

        self.setPen(QPen(QColor(255, 255, 255), 2))

        # Tooltip
        self.setToolTip(f"{port_name} ({port.data_type.value})")

        # Interactive - IMPORTANT: Enable mouse event handling
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)  # Don't select, just drag
        self.setFlag(QGraphicsItem.ItemIsFocusable, False)  # Don't take focus
        self.setAcceptedMouseButtons(Qt.LeftButton)

        # Set Z-value higher than parent node so ports are on top
        self.setZValue(10)

    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        self.setBrush(QBrush(QColor(255, 255, 100)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        if self.port_type == 'input':
            self.setBrush(QBrush(QColor(100, 150, 255)))
        else:
            self.setBrush(QBrush(QColor(255, 150, 100)))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        """Start dragging a connection from this port."""
        if event.button() == Qt.LeftButton:
            # Allow dragging from both input and output ports
            event.accept()

            # Create a temporary line to show while dragging
            self.temp_line = QGraphicsLineItem()
            self.temp_line.setPen(QPen(QColor(255, 255, 100), 2, Qt.DashLine))
            self.temp_line.setZValue(1000)  # Draw on top
            self.scene().addItem(self.temp_line)

            # Store starting position
            start_pos = self.get_scene_center()
            self.temp_line.setLine(QLineF(start_pos, start_pos))

            self.dragging_connection = True
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Update the temporary connection line while dragging."""
        if self.dragging_connection and self.temp_line:
            event.accept()

            # Update the line to follow the mouse
            start_pos = self.get_scene_center()
            end_pos = event.scenePos()
            self.temp_line.setLine(QLineF(start_pos, end_pos))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Complete the connection if dropped on a valid port."""
        if event.button() == Qt.LeftButton and self.dragging_connection:
            event.accept()

            # Remove temporary line
            if self.temp_line:
                self.scene().removeItem(self.temp_line)
                self.temp_line = None

            # Check if we're over a compatible port
            items = self.scene().items(event.scenePos())
            target_port = None

            for item in items:
                if isinstance(item, PortGraphicsItem) and item != self:
                    # Allow connection if ports are different types
                    if self.port_type == 'output' and item.port_type == 'input':
                        target_port = item
                        break
                    elif self.port_type == 'input' and item.port_type == 'output':
                        target_port = item
                        break

            # Create connection if valid target found
            if target_port:
                # Ensure connection goes from output to input
                if self.port_type == 'output':
                    self._create_connection(self, target_port)
                else:
                    self._create_connection(target_port, self)

            self.dragging_connection = False
        else:
            super().mouseReleaseEvent(event)

    def _create_connection(self, source_port, target_port):
        """Create a connection between source and target ports."""
        # Check if connection already exists
        for conn in source_port.connections:
            if (conn.source_port == source_port and conn.target_port == target_port) or \
               (conn.source_port == target_port and conn.target_port == source_port):
                print(f"Connection already exists between {source_port.port_name} and {target_port.port_name}")
                return

        # Check if data types are compatible
        if source_port.port.data_type != target_port.port.data_type:
            print(f"Cannot connect: incompatible types {source_port.port.data_type} -> {target_port.port.data_type}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "Incompatible Port Types",
                f"Cannot connect {source_port.port.data_type.value} to {target_port.port.data_type.value}"
            )
            return

        # Create visual connection
        connection = ConnectionGraphicsItem(source_port, target_port)
        self.scene().addItem(connection)

        # Create logical connection in the workflow
        source_node = source_port.parentItem().node
        target_node = target_port.parentItem().node

        try:
            # Connect in the workflow
            from ..core import Link
            link = Link(
                source=source_node.outputs[source_port.port_name],
                target=target_node.inputs[target_port.port_name]
            )

            # Store link reference (if workflow/editor needs it)
            if hasattr(self.scene(), 'editor'):
                if not hasattr(self.scene().editor, 'workflow_links'):
                    self.scene().editor.workflow_links = []
                self.scene().editor.workflow_links.append(link)

            print(f"Connected: {source_node.name}.{source_port.port_name} -> {target_node.name}.{target_port.port_name}")

        except Exception as e:
            print(f"Error creating connection: {e}")
            import traceback
            traceback.print_exc()
            # Remove visual connection if logical connection failed
            self.scene().removeItem(connection)

    def get_scene_center(self):
        """Get the center position in scene coordinates."""
        return self.parentItem().mapToScene(self.pos())

    def update_connections(self):
        """Update all connection lines."""
        for connection in self.connections:
            connection.update_position()


class ConnectionGraphicsItem(QGraphicsLineItem):
    """Visual representation of a connection between ports."""

    def __init__(self, source_port, target_port):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port

        # Style
        self.setPen(QPen(QColor(150, 150, 150), 2, Qt.SolidLine))
        self.setZValue(-1)  # Draw behind nodes

        # Register with ports
        source_port.connections.append(self)
        target_port.connections.append(self)

        self.update_position()

    def update_position(self):
        """Update the line position based on port positions."""
        source_pos = self.source_port.get_scene_center()
        target_pos = self.target_port.get_scene_center()

        # Create a curved line using a bezier path
        path = QPainterPath()
        path.moveTo(source_pos)

        # Calculate control points for curve
        dx = target_pos.x() - source_pos.x()
        dy = target_pos.y() - source_pos.y()

        ctrl1 = QPointF(source_pos.x() + dx * 0.5, source_pos.y())
        ctrl2 = QPointF(target_pos.x() - dx * 0.5, target_pos.y())

        path.cubicTo(ctrl1, ctrl2, target_pos)

        # Convert path to line for simple rendering
        # (In a full implementation, override paint() to draw the curve)
        self.setLine(QLineF(source_pos, target_pos))

    def paint(self, painter, option, widget):
        """Paint the connection with a curve."""
        painter.setRenderHint(QPainter.Antialiasing)

        source_pos = self.source_port.get_scene_center()
        target_pos = self.target_port.get_scene_center()

        # Create curved path
        path = QPainterPath()
        path.moveTo(source_pos)

        dx = target_pos.x() - source_pos.x()
        ctrl1 = QPointF(source_pos.x() + dx * 0.5, source_pos.y())
        ctrl2 = QPointF(target_pos.x() - dx * 0.5, target_pos.y())

        path.cubicTo(ctrl1, ctrl2, target_pos)

        # Draw the curve
        pen = QPen(QColor(150, 150, 150), 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawPath(path)

        # Draw arrowhead at target
        angle = math.atan2(target_pos.y() - ctrl2.y(), target_pos.x() - ctrl2.x())
        arrow_size = 8

        arrow_p1 = target_pos - QPointF(
            math.cos(angle + math.pi / 6) * arrow_size,
            math.sin(angle + math.pi / 6) * arrow_size
        )
        arrow_p2 = target_pos - QPointF(
            math.cos(angle - math.pi / 6) * arrow_size,
            math.sin(angle - math.pi / 6) * arrow_size
        )

        painter.setBrush(QBrush(QColor(150, 150, 150)))
        arrow_path = QPainterPath()
        arrow_path.moveTo(target_pos)
        arrow_path.lineTo(arrow_p1)
        arrow_path.lineTo(arrow_p2)
        arrow_path.closeSubpath()
        painter.drawPath(arrow_path)


class NodeParameterDialog(QDialog):
    """
    Dialog for editing node parameters.

    Dynamically creates form fields based on node's field definitions.
    """

    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self.field_widgets = {}  # Store widgets by field name

        self.setWindowTitle(f"Edit Parameters - {node.name}")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"<b>{self.node.__class__.__name__}</b>")
        title.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(title)

        # Get field definitions from node
        try:
            field_defs = self.node.get_field_definitions()
        except (AttributeError, NotImplementedError):
            field_defs = {}

        if not field_defs:
            # No configurable parameters
            no_params_label = QLabel("This node has no configurable parameters.")
            no_params_label.setStyleSheet("padding: 20px; color: #888;")
            layout.addWidget(no_params_label)
        else:
            # Create scroll area for parameters
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("""
                QScrollArea {
                    border: 1px solid #555;
                    border-radius: 5px;
                    background: #2A2A2A;
                }
            """)

            # Form widget
            form_widget = QWidget()
            form_layout = QFormLayout()
            form_layout.setSpacing(15)
            form_layout.setContentsMargins(15, 15, 15, 15)

            # Create widgets for each field
            for field_name, field_info in field_defs.items():
                field_type = field_info.get('type', 'text')
                label = field_info.get('label', field_name)
                default = field_info.get('default', '')

                # Get current value from node config
                current_value = self.node.config.get(field_name, default)

                # Create appropriate widget based on type
                if field_type == 'choice':
                    widget = QComboBox()
                    choices = field_info.get('choices', [])
                    widget.addItems([str(c) for c in choices])

                    # Set current value
                    try:
                        idx = widget.findText(str(current_value))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                    except:
                        pass

                    widget.setStyleSheet("""
                        QComboBox {
                            padding: 5px;
                            background: #353535;
                            border: 1px solid #555;
                            border-radius: 3px;
                            color: white;
                        }
                        QComboBox::drop-down {
                            border: none;
                        }
                        QComboBox::down-arrow {
                            image: none;
                            border-left: 5px solid transparent;
                            border-right: 5px solid transparent;
                            border-top: 5px solid #888;
                            margin-right: 5px;
                        }
                    """)

                else:  # text or number
                    widget = QLineEdit()
                    widget.setText(str(current_value))
                    widget.setStyleSheet("""
                        QLineEdit {
                            padding: 5px;
                            background: #353535;
                            border: 1px solid #555;
                            border-radius: 3px;
                            color: white;
                        }
                        QLineEdit:focus {
                            border: 1px solid #4A90E2;
                        }
                    """)

                # Add to form
                label_widget = QLabel(label + ":")
                label_widget.setStyleSheet("color: #CCC;")
                form_layout.addRow(label_widget, widget)

                # Store widget
                self.field_widgets[field_name] = widget

            form_widget.setLayout(form_layout)
            scroll.setWidget(form_widget)
            layout.addWidget(scroll)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background: #555;
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #666;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_values(self):
        """Get the values from all field widgets."""
        values = {}

        for field_name, widget in self.field_widgets.items():
            if isinstance(widget, QLineEdit):
                values[field_name] = widget.text()
            elif isinstance(widget, QComboBox):
                values[field_name] = widget.currentText()

        return values
