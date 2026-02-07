"""
Graph executor for running computational graphs.

The executor manages the execution of nodes in the correct order,
handles data transfer between nodes, and provides execution monitoring.
"""

from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import time
import traceback
from .graph import ComputationalGraph
from .node import BaseNode


class ExecutionStatus(Enum):
    """Status of execution."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionResult:
    """Result of graph execution."""

    def __init__(self):
        self.status = ExecutionStatus.NOT_STARTED
        self.node_results: Dict[str, Any] = {}
        self.node_status: Dict[str, ExecutionStatus] = {}
        self.node_errors: Dict[str, str] = {}
        self.execution_time: float = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def get_node_result(self, node_name: str) -> Optional[Any]:
        """Get the result from a specific node."""
        return self.node_results.get(node_name)

    def get_node_status(self, node_name: str) -> ExecutionStatus:
        """Get the execution status of a specific node."""
        return self.node_status.get(node_name, ExecutionStatus.NOT_STARTED)

    def is_successful(self) -> bool:
        """Check if execution completed successfully."""
        return self.status == ExecutionStatus.COMPLETED

    def get_errors(self) -> Dict[str, str]:
        """Get all node errors."""
        return self.node_errors.copy()

    def __repr__(self):
        return (f"ExecutionResult(status={self.status.value}, "
                f"nodes={len(self.node_results)}, "
                f"time={self.execution_time:.3f}s)")


class GraphExecutor:
    """
    Executor for running computational graphs.

    Handles node execution in topological order, data transfer between
    nodes, error handling, and execution monitoring.
    """

    def __init__(
        self,
        graph: ComputationalGraph,
        max_iterations: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize executor.

        Args:
            graph: The computational graph to execute
            max_iterations: Maximum execution iterations (default: 2 * num_nodes)
            progress_callback: Optional callback(node_name, progress) for monitoring
        """
        self.graph = graph
        self.max_iterations = max_iterations or (len(graph.nodes) * 2)
        self.progress_callback = progress_callback
        self._cancel_requested = False

    def execute(self) -> ExecutionResult:
        """
        Execute the computational graph.

        Returns:
            ExecutionResult containing status, results, and timing info
        """
        result = ExecutionResult()
        result.status = ExecutionStatus.RUNNING
        result.start_time = time.time()

        try:
            # Validate graph
            is_valid, errors = self.graph.validate()
            if not is_valid:
                raise ValueError(f"Graph validation failed: {'; '.join(errors)}")

            # Reset all nodes
            self.graph.reset()

            # Get execution order
            execution_order = self.graph.get_execution_order()

            # Initialize node status
            for node_name in execution_order:
                result.node_status[node_name] = ExecutionStatus.NOT_STARTED

            # Execute nodes in order
            for idx, node_name in enumerate(execution_order):
                if self._cancel_requested:
                    result.status = ExecutionStatus.CANCELLED
                    return result

                node = self.graph.get_node(node_name)
                if not node:
                    continue

                # Update progress
                progress = (idx + 1) / len(execution_order)
                if self.progress_callback:
                    self.progress_callback(node_name, progress)

                # Execute node
                try:
                    result.node_status[node_name] = ExecutionStatus.RUNNING

                    # Transfer data from connected inputs
                    for input_port in node.inputs.values():
                        for link in input_port.links:
                            link.transfer_data()

                    # Execute node
                    success = node.execute()

                    if success:
                        result.node_status[node_name] = ExecutionStatus.COMPLETED

                        # Store node results (output port values)
                        result.node_results[node_name] = {
                            port_name: port.get_value()
                            for port_name, port in node.outputs.items()
                        }
                    else:
                        result.node_status[node_name] = ExecutionStatus.FAILED
                        result.node_errors[node_name] = "Node execution returned False"

                except Exception as e:
                    result.node_status[node_name] = ExecutionStatus.FAILED
                    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    result.node_errors[node_name] = error_msg
                    print(f"Error executing node '{node_name}': {error_msg}")

                    # Decide whether to continue or abort
                    if self._should_abort_on_error(node, result):
                        result.status = ExecutionStatus.FAILED
                        break

            # Check overall status
            if result.status == ExecutionStatus.RUNNING:
                if any(s == ExecutionStatus.FAILED for s in result.node_status.values()):
                    result.status = ExecutionStatus.FAILED
                else:
                    result.status = ExecutionStatus.COMPLETED

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            error_msg = f"Graph execution failed: {type(e).__name__}: {str(e)}"
            result.node_errors['__graph__'] = error_msg
            print(error_msg)

        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time

        return result

    def execute_node(self, node_name: str) -> ExecutionResult:
        """
        Execute a single node and its dependencies.

        Args:
            node_name: Name of the node to execute

        Returns:
            ExecutionResult for the subgraph execution
        """
        # Find all dependencies
        dependencies = self._get_all_dependencies(node_name)
        dependencies.add(node_name)

        # Create subgraph
        subgraph = self.graph.get_subgraph(list(dependencies))

        # Execute subgraph
        executor = GraphExecutor(subgraph, self.max_iterations, self.progress_callback)
        return executor.execute()

    def _get_all_dependencies(self, node_name: str, visited: Optional[set] = None) -> set:
        """Recursively get all dependencies of a node."""
        if visited is None:
            visited = set()

        if node_name in visited:
            return visited

        visited.add(node_name)
        dependencies = self.graph.get_dependencies(node_name)

        for dep in dependencies:
            self._get_all_dependencies(dep, visited)

        return visited

    def _should_abort_on_error(self, node: BaseNode, result: ExecutionResult) -> bool:
        """
        Determine whether to abort execution after a node error.

        Override this method to customize error handling behavior.
        """
        # By default, abort on any error
        # Can be customized to continue execution for non-critical nodes
        return True

    def cancel(self):
        """Cancel the currently running execution."""
        self._cancel_requested = True

    def reset_cancel(self):
        """Reset cancel flag."""
        self._cancel_requested = False


class AsyncExecutor(GraphExecutor):
    """
    Asynchronous executor for parallel node execution.

    Executes independent nodes in parallel when possible.
    """

    def __init__(
        self,
        graph: ComputationalGraph,
        max_workers: int = 4,
        max_iterations: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize async executor.

        Args:
            graph: The computational graph to execute
            max_workers: Maximum number of parallel workers
            max_iterations: Maximum execution iterations
            progress_callback: Optional progress callback
        """
        super().__init__(graph, max_iterations, progress_callback)
        self.max_workers = max_workers

    def execute(self) -> ExecutionResult:
        """
        Execute graph with parallel execution of independent nodes.

        This is a placeholder implementation. Full async execution would
        require threading/multiprocessing support.
        """
        # For now, fall back to sequential execution
        # TODO: Implement true parallel execution
        return super().execute()


class InteractiveExecutor(GraphExecutor):
    """
    Interactive executor that allows step-by-step execution.

    Useful for debugging and visualization.
    """

    def __init__(
        self,
        graph: ComputationalGraph,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        super().__init__(graph, progress_callback=progress_callback)
        self.current_step = 0
        self.execution_order = []
        self._result: Optional[ExecutionResult] = None

    def start(self) -> ExecutionResult:
        """Start interactive execution."""
        self._result = ExecutionResult()
        self._result.status = ExecutionStatus.RUNNING
        self._result.start_time = time.time()

        # Validate and get execution order
        is_valid, errors = self.graph.validate()
        if not is_valid:
            self._result.status = ExecutionStatus.FAILED
            self._result.node_errors['__graph__'] = '; '.join(errors)
            return self._result

        self.graph.reset()
        self.execution_order = self.graph.get_execution_order()
        self.current_step = 0

        for node_name in self.execution_order:
            self._result.node_status[node_name] = ExecutionStatus.NOT_STARTED

        return self._result

    def step(self) -> tuple[bool, Optional[str]]:
        """
        Execute one step (one node).

        Returns:
            Tuple of (has_more_steps, current_node_name)
        """
        if not self._result or self.current_step >= len(self.execution_order):
            return False, None

        node_name = self.execution_order[self.current_step]
        node = self.graph.get_node(node_name)

        if node:
            try:
                self._result.node_status[node_name] = ExecutionStatus.RUNNING

                # Transfer input data
                for input_port in node.inputs.values():
                    for link in input_port.links:
                        link.transfer_data()

                # Execute
                success = node.execute()

                if success:
                    self._result.node_status[node_name] = ExecutionStatus.COMPLETED
                    self._result.node_results[node_name] = {
                        port_name: port.get_value()
                        for port_name, port in node.outputs.items()
                    }
                else:
                    self._result.node_status[node_name] = ExecutionStatus.FAILED

            except Exception as e:
                self._result.node_status[node_name] = ExecutionStatus.FAILED
                self._result.node_errors[node_name] = str(e)

        self.current_step += 1

        # Check if done
        if self.current_step >= len(self.execution_order):
            self._result.end_time = time.time()
            self._result.execution_time = self._result.end_time - self._result.start_time
            self._result.status = ExecutionStatus.COMPLETED

            return False, node_name

        return True, node_name

    def get_result(self) -> Optional[ExecutionResult]:
        """Get the current execution result."""
        return self._result
