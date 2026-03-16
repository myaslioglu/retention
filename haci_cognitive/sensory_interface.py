"""
🔌 Sensory Interface - Level 3 (Extensible Framework)
Plug-and-play interface for future sensors, actuators, and environment interaction.

Design Philosophy:
- Base classes for all sensory inputs and motor outputs
- Registry pattern for dynamic plugin loading
- Event-driven architecture
- Ready for: vision, audio, touch, motion, environment

Future Extensions (when needed):
- CameraInput (vision)
- AudioInput (microphone)
- TactileInput (touch sensors)
- RobotArmOutput (actuator)
- VoiceOutput (speech)
- EnvironmentAPI (home automation, IoT)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SensorType(Enum):
    VISION = "vision"
    AUDIO = "audio"
    TACTILE = "tactile"
    TEXT = "text"
    NUMERIC = "numeric"
    ENVIRONMENT = "environment"
    CUSTOM = "custom"


class ActuatorType(Enum):
    DISPLAY = "display"
    SPEAKER = "speaker"
    MOTION = "motion"
    TEXT_OUT = "text_out"
    API_CALL = "api_call"
    CUSTOM = "custom"


@dataclass
class SensoryInput:
    """A single sensory input event."""
    sensor_type: SensorType
    data: Any
    timestamp: str = ""
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MotorOutput:
    """A single motor/output command."""
    actuator_type: ActuatorType
    command: str
    parameters: Dict = field(default_factory=dict)
    priority: int = 5  # 1=urgent, 10=low
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SensoryProcessor(ABC):
    """Base class for sensory input processors."""
    
    @abstractmethod
    def process(self, input_data: Any) -> SensoryInput:
        """Process raw input into structured sensory input."""
        pass
    
    @abstractmethod
    def get_sensor_type(self) -> SensorType:
        """Return the sensor type."""
        pass
    
    def is_available(self) -> bool:
        """Check if this sensor is available."""
        return True


class MotorController(ABC):
    """Base class for motor/output controllers."""
    
    @abstractmethod
    def execute(self, output: MotorOutput) -> bool:
        """Execute a motor command."""
        pass
    
    @abstractmethod
    def get_actuator_type(self) -> ActuatorType:
        """Return the actuator type."""
        pass
    
    def is_available(self) -> bool:
        """Check if this actuator is available."""
        return True


class TextSensor(SensoryProcessor):
    """Text input sensor (always available)."""
    
    def process(self, input_data: Any) -> SensoryInput:
        return SensoryInput(
            sensor_type=SensorType.TEXT,
            data=str(input_data),
            confidence=1.0,
        )
    
    def get_sensor_type(self) -> SensorType:
        return SensorType.TEXT


class TextActuator(MotorController):
    """Text output actuator (always available)."""
    
    def __init__(self):
        self.output_buffer: List[str] = []
    
    def execute(self, output: MotorOutput) -> bool:
        text = output.parameters.get('text', output.command)
        self.output_buffer.append(text)
        logger.info(f"📝 Text output: {text[:50]}...")
        return True
    
    def get_actuator_type(self) -> ActuatorType:
        return ActuatorType.TEXT_OUT
    
    def get_output(self) -> List[str]:
        """Get all buffered outputs."""
        return self.output_buffer.copy()
    
    def clear(self):
        """Clear output buffer."""
        self.output_buffer.clear()


class SensoryInterface:
    """
    Main sensory interface registry.
    Manages sensors and actuators with plug-and-play architecture.
    """
    
    def __init__(self, state_dir: str = "cognitive_state"):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "sensory_state.json"
        
        # Registries
        self.sensors: Dict[str, SensoryProcessor] = {}
        self.actuators: Dict[str, MotorController] = {}
        
        # Event history
        self.input_history: List[Dict] = []
        self.output_history: List[Dict] = []
        
        # Event callbacks
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Register built-in sensors/actuators
        self._register_builtin()
        
        self.load_state()
        logger.info(f"🔌 SensoryInterface initialized ({len(self.sensors)} sensors, {len(self.actuators)} actuators)")
    
    def _register_builtin(self):
        """Register built-in sensors and actuators."""
        self.register_sensor('text', TextSensor())
        self.register_actuator('text', TextActuator())
    
    def register_sensor(self, name: str, sensor: SensoryProcessor):
        """Register a new sensor."""
        self.sensors[name] = sensor
        logger.info(f"🔌 Sensor registered: {name} ({sensor.get_sensor_type().value})")
    
    def register_actuator(self, name: str, actuator: MotorController):
        """Register a new actuator."""
        self.actuators[name] = actuator
        logger.info(f"🔌 Actuator registered: {name} ({actuator.get_actuator_type().value})")
    
    def unregister_sensor(self, name: str):
        """Unregister a sensor."""
        if name in self.sensors:
            del self.sensors[name]
            logger.info(f"🔌 Sensor unregistered: {name}")
    
    def unregister_actuator(self, name: str):
        """Unregister an actuator."""
        if name in self.actuators:
            del self.actuators[name]
            logger.info(f"🔌 Actuator unregistered: {name}")
    
    def receive_input(self, sensor_name: str, data: Any) -> Optional[SensoryInput]:
        """Receive input from a named sensor."""
        sensor = self.sensors.get(sensor_name)
        if not sensor:
            logger.warning(f"Sensor '{sensor_name}' not found")
            return None
        
        if not sensor.is_available():
            logger.warning(f"Sensor '{sensor_name}' not available")
            return None
        
        sensory_input = sensor.process(data)
        
        # Record
        self.input_history.append({
            'sensor': sensor_name,
            'type': sensory_input.sensor_type.value,
            'timestamp': sensory_input.timestamp,
            'confidence': sensory_input.confidence,
        })
        
        # Trigger event handlers
        self._trigger_event(f"input_{sensory_input.sensor_type.value}", sensory_input)
        
        return sensory_input
    
    def send_output(self, actuator_name: str, command: str, **kwargs) -> bool:
        """Send output to a named actuator."""
        actuator = self.actuators.get(actuator_name)
        if not actuator:
            logger.warning(f"Actuator '{actuator_name}' not found")
            return False
        
        if not actuator.is_available():
            logger.warning(f"Actuator '{actuator_name}' not available")
            return False
        
        output = MotorOutput(
            actuator_type=actuator.get_actuator_type(),
            command=command,
            parameters=kwargs,
        )
        
        success = actuator.execute(output)
        
        # Record
        self.output_history.append({
            'actuator': actuator_name,
            'command': command,
            'success': success,
            'timestamp': output.timestamp,
        })
        
        return success
    
    def on_event(self, event_type: str, handler: Callable):
        """Register an event handler."""
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: str, data: Any):
        """Trigger event handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_available_sensors(self) -> List[str]:
        """List available sensors."""
        return [name for name, s in self.sensors.items() if s.is_available()]
    
    def get_available_actuators(self) -> List[str]:
        """List available actuators."""
        return [name for name, a in self.actuators.items() if a.is_available()]
    
    def get_status(self) -> Dict:
        """Get interface status."""
        return {
            'sensors': {
                name: {
                    'type': s.get_sensor_type().value,
                    'available': s.is_available(),
                } for name, s in self.sensors.items()
            },
            'actuators': {
                name: {
                    'type': a.get_actuator_type().value,
                    'available': a.is_available(),
                } for name, a in self.actuators.items()
            },
            'total_inputs': len(self.input_history),
            'total_outputs': len(self.output_history),
        }
    
    def save_state(self):
        """Save state."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            'registered_sensors': list(self.sensors.keys()),
            'registered_actuators': list(self.actuators.keys()),
            'input_history': self.input_history[-100:],
            'output_history': self.output_history[-100:],
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self):
        """Load state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.input_history = state.get('input_history', [])
                self.output_history = state.get('output_history', [])
                logger.info(f"📂 Loaded sensory interface state")
            except Exception as e:
                logger.warning(f"Failed to load sensory state: {e}")
    
    def get_summary(self) -> str:
        """Summary."""
        lines = [
            "🔌 Sensory Interface",
            "=" * 40,
            f"Sensors: {len(self.sensors)} ({', '.join(self.get_available_sensors())})",
            f"Actuators: {len(self.actuators)} ({', '.join(self.get_available_actuators())})",
            f"Total inputs: {len(self.input_history)}",
            f"Total outputs: {len(self.output_history)}",
            "",
            "🔌 Extension Points (future):",
            "  - CameraInput → vision processing",
            "  - AudioInput → sound recognition",
            "  - TactileInput → touch feedback",
            "  - RobotArmOutput → physical action",
            "  - VoiceOutput → speech synthesis",
            "  - EnvironmentAPI → IoT/home control",
        ]
        return "\n".join(lines)


# ============================================================
# Example extension template (for future use)
# ============================================================

class ExampleCameraSensor(SensoryProcessor):
    """
    Example: Camera sensor for future vision processing.
    Uncomment and implement when hardware is available.
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        # self.camera = cv2.VideoCapture(camera_id)  # Future
    
    def process(self, input_data: Any = None) -> SensoryInput:
        # frame = self.camera.read()
        # return SensoryInput(sensor_type=SensorType.VISION, data=frame, confidence=0.95)
        raise NotImplementedError("Camera hardware not connected")
    
    def get_sensor_type(self) -> SensorType:
        return SensorType.VISION
    
    def is_available(self) -> bool:
        return False  # Change to True when hardware is connected


class ExampleVoiceActuator(MotorController):
    """
    Example: Voice output actuator.
    Uncomment and implement when TTS is available.
    """
    
    def execute(self, output: MotorOutput) -> bool:
        text = output.parameters.get('text', output.command)
        # tts_engine.speak(text)  # Future
        logger.info(f"🗣️ Would say: {text}")
        return True
    
    def get_actuator_type(self) -> ActuatorType:
        return ActuatorType.SPEAKER
    
    def is_available(self) -> bool:
        return True  # Soft-available (can log but not actually speak)
