"""
Mock modules for handling dependencies that may not be available.
"""

import sys


class MockThread:
    """Mock thread class for PySide6 compatibility."""
    
    def __init__(self):
        self.node = MockNode()


class MockNode:
    """Mock node class for PySide6 compatibility."""
    
    def __init__(self):
        self.id = "mock_node_id"


class MockGlobalStop:
    """Mock global stop class for workflow compatibility."""
    
    @property
    def stop_requested_threads(self) -> bool:
        return False


class MockQThread:
    """Mock QThread class for PySide6 compatibility."""
    
    @staticmethod
    def currentThread() -> MockThread:
        return MockThread()


def setup_mock_modules() -> None:
    """Set up mock modules for dependencies that may not be available."""
    
    # Mock Workflow module
    sys.modules['Workflow.work_graph.work_graph_ui.module'] = type('MockModule', (), {
        'global_stop': MockGlobalStop()
    })
    
    # Mock PySide6 module
    sys.modules['PySide6'] = type('MockPySide6', (), {
        'QtCore': type('MockQtCore', (), {
            'QThread': MockQThread,
            'QThreadPool': type('MockQThreadPool', (), {}),
            'QProcess': type('MockQProcess', (), {}),
            'Slot': lambda *args, **kwargs: lambda f: f,
            'QPointF': type('MockQPointF', (), {})
        })
    }) 