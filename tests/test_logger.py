"""Quick test script for logger functionality."""

from src.utils.logger import setup_logger, get_logger, configure_from_dict
from src.utils.config import Config

# Test 1: Basic setup
print("=" * 60)
print("Test 1: Basic logger setup")
print("=" * 60)
setup_logger(level="DEBUG", log_file="logs/test.log")
log = get_logger(__name__)
log.debug("This is a debug message")
log.info("This is an info message")
log.warning("This is a warning message")
log.error("This is an error message")

# Test 2: Load from config file
print("\n" + "=" * 60)
print("Test 2: Logger from config file")
print("=" * 60)
try:
    config = Config()
    # Convert LoggingConfig dataclass to dict
    logging_config = {
        "level": config.logging.level,
        "format": config.logging.format,
        "file": config.logging.file,
        "rotation": config.logging.rotation,
        "retention": config.logging.retention,
    }
    configure_from_dict(logging_config)
    log2 = get_logger("test_module")
    log2.info("Logger configured from config.yaml")
    log2.debug("Debug message (may not show if level is INFO)")
    log2.warning("Warning message with context")
    print("\n✓ Logger configuration successful!")
except Exception as e:
    print(f"\n✗ Error loading config: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Check logs/test.log and logs/discovery.log for output")
print("=" * 60)
