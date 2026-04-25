"""
Enhanced Multi-PINNACLE Core Module
====================================

Core consciousness system and frameworks for the Enhanced Multi-PINNACLE system.

Main Components:
- Enhanced Multi-PINNACLE System: Complete integrated consciousness system
- Consciousness Frameworks: Individual framework implementations
- Reasoning Engines: Advanced reasoning capabilities
- Integration Systems: Framework merger and state management

Usage:
    from core import EnhancedMultiPinnacleSystem, create_enhanced_system
    
    # Create system with default configuration
    system = create_enhanced_system()
    
    # Solve ARC problem
    results = system.solve_arc_problem(problem_data)
"""

from .enhanced_multi_pinnacle import (
    EnhancedMultiPinnacleSystem,
    EnhancedMultiPinnacleConfig,
    SystemPerformanceMetrics,
    create_enhanced_system
)

# Import consciousness frameworks
try:
    from .consciousness_frameworks import (
        UniversalMindGenerator,
        ThreePrinciplesFramework,
        DeschoolingSocietyIntegration,
        TranscendentStatesProcessor,
        HRMCyclesManager
    )
except ImportError:
    # Graceful fallback if not all frameworks are available
    pass

__all__ = [
    'EnhancedMultiPinnacleSystem',
    'EnhancedMultiPinnacleConfig', 
    'SystemPerformanceMetrics',
    'create_enhanced_system',
]

__version__ = "1.0.0"
__author__ = "Enhanced Multi-PINNACLE Team"
__email__ = "contact@enhanced-multi-pinnacle.ai"