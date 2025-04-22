import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""

    pass


@dataclass
class Model:
    """Represents a machine learning model configuration."""

    id: str
    type: str
    version: str = "1.0.0"
    accuracy: float = 0.0
    path: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, model_id: str, config_item: Dict[str, Any]) -> "Model":
        """Create a Model from a config dictionary item"""
        return cls(
            id=model_id,
            type=config_item.get("type", "unknown"),
            version=config_item.get("version", "1.0.0"),
            accuracy=config_item.get("accuracy", 0.0),
            path=config_item.get("path", ""),
            description=config_item.get("description", ""),
            parameters=config_item.get("parameters", {}),
        )


@dataclass
class Interface:
    """Represents a user interface configuration."""

    id: str
    description: str
    default_problem: str = ""
    final: bool = False

    @classmethod
    def from_config(cls, config_item: Dict[str, Any]) -> "Interface":
        """Create an Interface from a config dictionary item"""
        return cls(
            id=config_item["id"],
            description=config_item["description"],
            default_problem=config_item.get("default_problem", ""),
            final=config_item.get("final", False),
        )


@dataclass
class ValidCombination:
    """Represents a valid combination of interface and problems."""

    interface: str
    problems: List[str] = field(default_factory=list)
    final: bool = False

    @classmethod
    def from_config(cls, config_item: Dict[str, Any]) -> "ValidCombination":
        """Create a ValidCombination from a config dictionary item"""
        return cls(
            interface=config_item["interface"],
            problems=config_item.get("problems", []),
            final=config_item.get("final", False),
        )


class ConfigValidator:
    """Validates configuration files for the Tulip application."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the validator with the configuration directory.

        Args:
            config_dir: Path to the configuration directory. If None, tries to detect it.
        """
        if config_dir is None:
            # Try to locate the config directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_config_dirs = [
                # When running from packages/server/src
                os.path.abspath(os.path.join(script_dir, "..", "..", "..", "config")),
                # When running from project root
                os.path.abspath(os.path.join(script_dir, "config")),
            ]

            for path in possible_config_dirs:
                if os.path.isdir(path) and os.path.exists(
                    os.path.join(path, "master.yaml")
                ):
                    config_dir = path
                    break

            if config_dir is None:
                raise ConfigValidationError(
                    "Could not locate config directory. Please specify the path explicitly."
                )

        self.config_dir = Path(config_dir)
        self.master_config_path = self.config_dir / "master.yaml"
        self.problems_dir = self.config_dir / "problems"
        self.interfaces_dir = self.config_dir / "interfaces"
        self.models_dir = self.config_dir / "models"  # New directory for model configs

        # Ensure required directories exist
        if not self.config_dir.exists():
            raise ConfigValidationError(
                f"Config directory does not exist: {self.config_dir}"
            )
        if not self.master_config_path.exists():
            raise ConfigValidationError(
                f"Master config file not found: {self.master_config_path}"
            )
        if not self.problems_dir.exists():
            raise ConfigValidationError(
                f"Problems directory not found: {self.problems_dir}"
            )

        # Interfaces directory might not exist in older versions
        if not self.interfaces_dir.exists():
            os.makedirs(self.interfaces_dir, exist_ok=True)

        # Models directory might not exist yet
        if not self.models_dir.exists():
            os.makedirs(self.models_dir, exist_ok=True)

    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML content

        Raises:
            ConfigValidationError: If the file cannot be loaded or parsed
        """
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigValidationError(f"Error loading {file_path}: {str(e)}")

    def validate_master_config(self) -> Dict[str, Any]:
        """
        Validate the master configuration file.

        Returns:
            The validated master configuration

        Raises:
            ConfigValidationError: If validation fails
        """
        master_config = self.load_yaml(self.master_config_path)

        # Check required sections
        required_sections = ["interfaces", "problems"]
        for section in required_sections:
            if section not in master_config:
                raise ConfigValidationError(
                    f"Master config is missing required section: {section}"
                )

            if not isinstance(master_config[section], list):
                raise ConfigValidationError(
                    f"Master config section '{section}' must be a list"
                )

        # Validate interfaces
        for i, interface in enumerate(master_config.get("interfaces", [])):
            self._validate_interface_entry(interface, i)

        # Validate problems
        for i, problem in enumerate(master_config.get("problems", [])):
            self._validate_problem_entry(problem, i)

        # Validate models if present
        if "models" in master_config:
            if not isinstance(master_config["models"], list):
                raise ConfigValidationError(
                    "Master config section 'models' must be a list"
                )

            for i, model in enumerate(master_config["models"]):
                self._validate_model_entry(model, i)

        # Validate valid_combinations if present
        if "valid_combinations" in master_config:
            if not isinstance(master_config["valid_combinations"], list):
                raise ConfigValidationError("valid_combinations must be a list")

            for i, combo in enumerate(master_config["valid_combinations"]):
                if not isinstance(combo, dict):
                    raise ConfigValidationError(f"Combination {i} must be a dictionary")

                if "interface" not in combo:
                    raise ConfigValidationError(
                        f"Combination {i} is missing required field: interface"
                    )

                if "problems" not in combo:
                    raise ConfigValidationError(
                        f"Combination {i} is missing required field: problems"
                    )

                if not isinstance(combo["problems"], list):
                    raise ConfigValidationError(
                        f"Combination {i} 'problems' field must be a list"
                    )

                # Check that interface exists
                interface_id = combo["interface"]
                if not any(
                    i["id"] == interface_id for i in master_config.get("interfaces", [])
                ):
                    raise ConfigValidationError(
                        f"Combination {i} references unknown interface: {interface_id}"
                    )

                # Check that all problems exist
                for problem_id in combo["problems"]:
                    if not any(
                        p["id"] == problem_id for p in master_config.get("problems", [])
                    ):
                        raise ConfigValidationError(
                            f"Combination {i} references unknown problem: {problem_id}"
                        )

        return master_config

    def _validate_interface_entry(self, interface: Dict[str, Any], index: int) -> None:
        """Validate an interface entry in the master config."""
        required_fields = ["id", "description"]
        self._validate_required_fields(interface, required_fields, f"Interface {index}")

        # Validate interface id format (simple alphanumeric check)
        if not interface["id"].isalnum():
            raise ConfigValidationError(
                f"Interface id '{interface['id']}' must be alphanumeric"
            )

    def _validate_problem_entry(self, problem: Dict[str, Any], index: int) -> None:
        """Validate a problem entry in the master config."""
        required_fields = ["id", "description"]
        self._validate_required_fields(problem, required_fields, f"Problem {index}")

    def _validate_model_entry(self, model: Dict[str, Any], index: int) -> None:
        """Validate a model entry in the master config."""
        required_fields = ["id", "type"]
        self._validate_required_fields(model, required_fields, f"Model {index}")

    def _validate_required_fields(
        self, item: Dict[str, Any], required_fields: List[str], item_name: str
    ) -> None:
        """Check if all required fields are present in an item."""
        for field in required_fields:
            if field not in item:
                raise ConfigValidationError(
                    f"{item_name} is missing required field: {field}"
                )

            if not item[field]:  # Check for empty values
                raise ConfigValidationError(
                    f"{item_name} has empty value for field: {field}"
                )

    def validate_interface_config(self, interface_id: str) -> Dict[str, Any]:
        """
        Validate an interface configuration file.

        Args:
            interface_id: The ID of the interface to validate

        Returns:
            The validated interface configuration

        Raises:
            ConfigValidationError: If validation fails
        """
        interface_path = self.interfaces_dir / f"{interface_id}.yaml"
        if not interface_path.exists():
            # Not all interfaces may have a config file, this is acceptable
            return {
                "interface": {
                    "id": interface_id,
                    "description": "",
                }
            }

        interface_config = self.load_yaml(interface_path)

        # Check for required sections
        if "interface" not in interface_config and "dashboard" not in interface_config:
            raise ConfigValidationError(
                f"Interface config {interface_id} is missing 'interface' or 'dashboard' section"
            )

        # Check for either interface or dashboard section (for backward compatibility)
        if "interface" in interface_config:
            section_name = "interface"
        else:
            section_name = "dashboard"  # For backward compatibility

        interface = interface_config[section_name]
        required_fields = ["id", "description"]
        self._validate_required_fields(
            interface, required_fields, f"Interface {interface_id}"
        )

        # Validate layout if present
        if "layout" in interface:
            if not isinstance(interface["layout"], dict):
                raise ConfigValidationError(
                    f"Interface {interface_id} layout must be a dictionary"
                )

        # Validate panels if present
        if "panels" in interface_config:
            if not isinstance(interface_config["panels"], list):
                raise ConfigValidationError(
                    f"Interface {interface_id} 'panels' must be a list"
                )

            for i, panel in enumerate(interface_config["panels"]):
                if not isinstance(panel, dict):
                    raise ConfigValidationError(
                        f"Interface {interface_id} panel {i} must be a dictionary"
                    )

                required_panel_fields = ["id", "title", "type"]
                self._validate_required_fields(
                    panel,
                    required_panel_fields,
                    f"Panel {i} in interface {interface_id}",
                )

        return interface_config

    def validate_problem_config(self, problem_id: str) -> Dict[str, Any]:
        """
        Validate a problem configuration file.

        Args:
            problem_id: The ID of the problem to validate

        Returns:
            The validated problem configuration

        Raises:
            ConfigValidationError: If validation fails
        """
        problem_path = self.problems_dir / f"{problem_id}.yaml"
        if not problem_path.exists():
            raise ConfigValidationError(f"Problem config not found: {problem_path}")

        problem_config = self.load_yaml(problem_path)

        # Check for required sections
        if "problem" not in problem_config:
            raise ConfigValidationError(
                f"Problem config {problem_id} is missing 'problem' section"
            )

        problem = problem_config["problem"]
        required_fields = ["id", "description"]
        self._validate_required_fields(
            problem, required_fields, f"Problem {problem_id}"
        )

        # Validate features if present
        if "features" in problem_config:
            if not isinstance(problem_config["features"], dict):
                raise ConfigValidationError(
                    f"Problem {problem_id} 'features' must be a dictionary"
                )

            for feature_name, feature in problem_config["features"].items():
                if not isinstance(feature, dict):
                    raise ConfigValidationError(
                        f"Problem {problem_id} feature '{feature_name}' must be a dictionary"
                    )

                if "type" not in feature:
                    raise ConfigValidationError(
                        f"Problem {problem_id} feature '{feature_name}' is missing 'type'"
                    )

        # Validate model if present
        if "model" in problem_config:
            if not isinstance(problem_config["model"], dict):
                raise ConfigValidationError(
                    f"Problem {problem_id} 'model' must be a dictionary"
                )

            model = problem_config["model"]
            if "type" not in model:
                raise ConfigValidationError(
                    f"Problem {problem_id} model is missing 'type'"
                )

        return problem_config

    def validate_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        Validate a model configuration file if it exists.

        Args:
            model_id: The ID of the model to validate

        Returns:
            The validated model configuration

        Raises:
            ConfigValidationError: If validation fails
        """
        # Check in the dedicated models directory
        model_path = self.models_dir / f"{model_id}.yaml"

        # If model file doesn't exist, it might be defined inline in master.yaml
        if not model_path.exists():
            master_config = self.load_yaml(self.master_config_path)
            models = master_config.get("models", [])
            for model in models:
                if model.get("id") == model_id:
                    # Return the inline model configuration
                    return {"model": model}

            # If we get here, the model was not found in either location
            return {"model": {"id": model_id, "type": "unknown"}}

        model_config = self.load_yaml(model_path)

        # Check for required sections
        if "model" not in model_config:
            raise ConfigValidationError(
                f"Model config {model_id} is missing 'model' section"
            )

        model = model_config["model"]
        required_fields = ["type"]
        self._validate_required_fields(model, required_fields, f"Model {model_id}")

        return model_config

    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all configuration files.

        Returns:
            A dictionary with validation results

        Raises:
            ConfigValidationError: If validation fails
        """
        results = {
            "master": None,
            "interfaces": {},
            "problems": {},
            "models": {},
            "valid_combinations": [],
            "errors": [],
        }

        # Validate master config
        try:
            master_config = self.validate_master_config()
            results["master"] = master_config
        except ConfigValidationError as e:
            results["errors"].append(f"Master config error: {str(e)}")
            return results

        # Validate all interfaces
        for interface in master_config.get("interfaces", []):
            interface_id = interface["id"]
            try:
                interface_config = self.validate_interface_config(interface_id)
                results["interfaces"][interface_id] = interface_config
            except ConfigValidationError as e:
                results["errors"].append(f"Interface '{interface_id}' error: {str(e)}")

        # Validate all problems
        for problem in master_config.get("problems", []):
            problem_id = problem["id"]
            try:
                problem_config = self.validate_problem_config(problem_id)
                results["problems"][problem_id] = problem_config
            except ConfigValidationError as e:
                results["errors"].append(f"Problem '{problem_id}' error: {str(e)}")

        # Validate all models
        for model in master_config.get("models", []):
            model_id = model["id"]
            try:
                model_config = self.validate_model_config(model_id)
                results["models"][model_id] = model_config
            except ConfigValidationError as e:
                results["errors"].append(f"Model '{model_id}' error: {str(e)}")

        # Also check for models defined in problem configs
        for problem_id, problem_config in results["problems"].items():
            if "model" in problem_config:
                model_data = problem_config["model"]
                if isinstance(model_data, dict) and "type" in model_data:
                    # Create a model_id if not explicitly defined
                    model_id = model_data.get("id", f"{problem_id}_model")
                    # Add to models collection if not already there
                    if model_id not in results["models"]:
                        results["models"][model_id] = {"model": model_data}

        # Extract valid combinations
        for combo in master_config.get("valid_combinations", []):
            results["valid_combinations"].append(ValidCombination.from_config(combo))

        return results

    def get_problems(self, validation_results=None) -> List[Problem]:
        """
        Get a list of Problem objects from the configuration.

        Args:
            validation_results: Optional validation results from validate_all()

        Returns:
            List of Problem objects
        """
        if validation_results is None:
            validation_results = self.validate_all()

        master_config = validation_results.get("master")
        if not master_config:
            return []

        problems = []
        for problem_config in master_config.get("problems", []):
            problem = Problem.from_config(problem_config)

            # Load detailed configuration if available
            problem_id = problem.id
            if problem_id in validation_results["problems"]:
                details = validation_results["problems"][problem_id]
                if "features" in details:
                    problem.features = details["features"]

                # Load model information
                if "model" in details:
                    model_data = details["model"]
                    model_id = model_data.get("id", f"{problem_id}_model")
                    problem.model = Model.from_config(model_id, model_data)

                # Check if this problem has a model referenced in master.yaml
                master_models = master_config.get("models", [])
                for model in master_models:
                    # If there's a model with matching problem_id reference or matching id
                    if (
                        model.get("problem_id") == problem_id
                        or model.get("id") == problem_id
                    ):
                        problem.model = Model.from_config(
                            model.get("id", model_id), model
                        )
                        break

            problems.append(problem)

        return problems

    def get_interfaces(self, validation_results=None) -> List[Interface]:
        """
        Get a list of Interface objects from the configuration.

        Args:
            validation_results: Optional validation results from validate_all()

        Returns:
            List of Interface objects
        """
        if validation_results is None:
            validation_results = self.validate_all()

        master_config = validation_results.get("master")
        if not master_config:
            return []

        interfaces = []
        for interface_config in master_config.get("interfaces", []):
            interface = Interface.from_config(interface_config)
            interfaces.append(interface)

        return interfaces

    def get_models(self, validation_results=None) -> List[Model]:
        """
        Get a list of Model objects from the configuration.

        Args:
            validation_results: Optional validation results from validate_all()

        Returns:
            List of Model objects
        """
        if validation_results is None:
            validation_results = self.validate_all()

        master_config = validation_results.get("master")
        if not master_config:
            return []

        models = []

        # First check models section in master config
        for model_config in master_config.get("models", []):
            model_id = model_config["id"]
            model = Model.from_config(model_id, model_config)
            models.append(model)

        # Then check models in individual problem configs
        for problem_id, problem_config in validation_results["problems"].items():
            if "model" in problem_config:
                model_data = problem_config["model"]
                if isinstance(model_data, dict) and "type" in model_data:
                    # Create a model_id if not explicitly defined
                    model_id = model_data.get("id", f"{problem_id}_model")
                    # Only add if not already added from master config
                    if not any(m.id == model_id for m in models):
                        model = Model.from_config(model_id, model_data)
                        models.append(model)

        return models

    def get_valid_combinations(self, validation_results=None) -> List[ValidCombination]:
        """
        Get a list of ValidCombination objects from the configuration.

        Args:
            validation_results: Optional validation results from validate_all()

        Returns:
            List of ValidCombination objects
        """
        if validation_results is None:
            validation_results = self.validate_all()

        return validation_results.get("valid_combinations", [])

    def is_valid_combination(
        self, interface_id: str, problem_id: str, validation_results=None
    ) -> bool:
        """
        Check if a interface-problem combination is valid.

        Args:
            interface_id: The ID of the interface
            problem_id: The ID of the problem
            validation_results: Optional validation results from validate_all()

        Returns:
            True if the combination is valid, False otherwise
        """
        if validation_results is None:
            validation_results = self.validate_all()

        valid_combinations = validation_results.get("valid_combinations", [])

        # If no valid combinations are specified, all combinations are valid
        if not valid_combinations:
            return True

        # Check if the combination is explicitly allowed
        for combo in valid_combinations:
            if combo.interface == interface_id and problem_id in combo.problems:
                return True

        return False


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    if results["errors"]:
        print("❌ Configuration validation failed with the following errors:")
        for error in results["errors"]:
            print(f"  - {error}")
        return

    print("✅ All configuration files are valid!")
    print(f"  - Master config: OK")
    print(f"  - Interfaces: {len(results['interfaces'])} valid")
    print(f"  - Problems: {len(results['problems'])} valid")
    print(f"  - Models: {len(results['models'])} valid")
    print(f"  - Valid Combinations: {len(results['valid_combinations'])}")


def main():
    """Run the validator as a standalone tool."""
    # Allow specifying config directory from command line
    config_dir = None
    if len(sys.argv) > 1:
        config_dir = sys.argv[1]

    try:
        validator = ConfigValidator(config_dir)
        results = validator.validate_all()
        print_validation_report(results)

        # Exit with error code if there were errors
        if results["errors"]:
            sys.exit(1)

    except ConfigValidationError as e:
        print(f"❌ Configuration validation error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
