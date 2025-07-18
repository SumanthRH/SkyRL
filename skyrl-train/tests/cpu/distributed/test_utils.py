import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from skyrl_train.distributed.utils import create_optimizer, register_optimizer, OPTIMIZER_REGISTRY


class MockConfig:
    """Mock configuration object for testing optimizer creation."""
    
    def __init__(self, optimizer_type="adamw", lr=1e-4, muon_lr=0.02, adam_betas=(0.9, 0.999), weight_decay=1e-2):
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.muon_lr = muon_lr
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay


def create_test_model():
    """Create a simple test model with different parameter types."""
    model = nn.Sequential(
        nn.Linear(10, 20),  # 2D parameters (weight + 1D bias)
        nn.LayerNorm(20),   # 1D parameters (weight, bias)
        nn.Linear(20, 5),   # 2D parameters (weight + 1D bias)
    )
    return model


class TestCreateOptimizer:
    """Test suite for the create_optimizer factory function."""

    def test_adamw_optimizer_creation(self):
        """Test that AdamW optimizer is created correctly."""
        model = create_test_model()
        config = MockConfig(optimizer_type="adamw")
        
        optimizer = create_optimizer("adamw", model.parameters(), config)
        
        # Verify optimizer type
        assert isinstance(optimizer, torch.optim.AdamW)
        
        # Verify configuration
        assert optimizer.param_groups[0]["lr"] == config.lr
        assert optimizer.param_groups[0]["betas"] == config.adam_betas
        assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
        
        # Verify all parameters are included
        optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
        model_params = list(model.parameters())
        assert len(optimizer_params) == len(model_params)

    def test_muon_optimizer_creation(self):
        """Test that Muon optimizer is created correctly with parameter separation."""
        model = create_test_model()
        config = MockConfig(optimizer_type="muon")
        
        # Skip test if Muon is not available
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            pytest.skip("Muon optimizer not available")
        
        optimizer = create_optimizer("muon", model.parameters(), config)
        
        # Verify optimizer type
        assert isinstance(optimizer, MuonWithAuxAdam)
        
        # Verify we have the expected number of parameter groups
        assert len(optimizer.param_groups) == 2
        
        # Separate expected parameters
        model_params = list(model.parameters())
        expected_2d_params = [p for p in model_params if p.ndim >= 2]
        expected_1d_params = [p for p in model_params if p.ndim < 2]
        
        # Find Muon and AdamW parameter groups
        muon_group = None
        adamw_group = None
        
        for group in optimizer.param_groups:
            if group.get("use_muon", False):
                muon_group = group
            else:
                adamw_group = group
        
        # Verify Muon group
        assert muon_group is not None
        assert muon_group["use_muon"] is True
        assert muon_group["lr"] == config.muon_lr
        assert muon_group["weight_decay"] == config.weight_decay
        assert len(muon_group["params"]) == len(expected_2d_params)
        
        # Verify AdamW group
        assert adamw_group is not None
        assert adamw_group.get("use_muon", True) is False  # use_muon should be False or not present
        assert adamw_group["lr"] == config.lr
        assert adamw_group["betas"] == config.adam_betas
        assert adamw_group["weight_decay"] == config.weight_decay
        assert len(adamw_group["params"]) == len(expected_1d_params)

    def test_muon_parameter_separation(self):
        """Test that parameters are correctly separated for Muon optimizer."""
        model = create_test_model()
        config = MockConfig(optimizer_type="muon")
        
        # Skip test if Muon is not available
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            pytest.skip("Muon optimizer not available")
        
        optimizer = create_optimizer("muon", model.parameters(), config)
        
        # Collect all parameters from both groups
        all_optimizer_params = []
        muon_params = []
        adamw_params = []
        
        for group in optimizer.param_groups:
            all_optimizer_params.extend(group["params"])
            if group.get("use_muon", False):
                muon_params.extend(group["params"])
            else:
                adamw_params.extend(group["params"])
        
        # Verify all model parameters are included exactly once
        model_params = list(model.parameters())
        assert len(all_optimizer_params) == len(model_params)
        assert set(id(p) for p in all_optimizer_params) == set(id(p) for p in model_params)
        
        # Verify parameter dimensions
        for param in muon_params:
            assert param.ndim >= 2, f"Muon group contains parameter with {param.ndim} dimensions"
        
        for param in adamw_params:
            assert param.ndim < 2, f"AdamW group contains parameter with {param.ndim} dimensions"

    def test_unknown_optimizer_error(self):
        """Test that unknown optimizer types raise appropriate errors."""
        model = create_test_model()
        config = MockConfig(optimizer_type="unknown")
        
        with pytest.raises(ValueError, match="Unknown optimizer type: unknown"):
            create_optimizer("unknown", model.parameters(), config)
        
        # Verify the error message includes available optimizers
        try:
            create_optimizer("unknown", model.parameters(), config)
        except ValueError as e:
            error_message = str(e)
            assert "Available optimizers:" in error_message
            assert "adamw" in error_message

    def test_empty_parameters_error(self):
        """Test that empty parameter list raises appropriate error."""
        config = MockConfig(optimizer_type="muon")
        
        # Skip test if Muon is not available
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            pytest.skip("Muon optimizer not available")
        
        with pytest.raises(ValueError, match="No parameters found for optimization"):
            create_optimizer("muon", [], config)

    def test_config_attribute_access(self):
        """Test that config attributes are accessed correctly."""
        model = create_test_model()
        
        # Test with missing muon_lr attribute (should use default)
        config_no_muon_lr = SimpleNamespace(
            lr=1e-4,
            adam_betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        
        # Skip test if Muon is not available
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            pytest.skip("Muon optimizer not available")
        
        optimizer = create_optimizer("muon", model.parameters(), config_no_muon_lr)
        
        # Find Muon group and verify default learning rate
        muon_group = next(group for group in optimizer.param_groups if group.get("use_muon", False))
        assert muon_group["lr"] == 0.02  # default muon_lr

    def test_backward_compatibility(self):
        """Test that default optimizer type falls back to AdamW."""
        model = create_test_model()
        config = MockConfig()  # No optimizer_type specified, should default to adamw
        
        # Test the function directly without optimizer_type in config
        config_dict = {"lr": 1e-4, "adam_betas": (0.9, 0.999), "weight_decay": 1e-2}
        config_obj = SimpleNamespace(**config_dict)
        
        optimizer = create_optimizer("adamw", model.parameters(), config_obj)
        assert isinstance(optimizer, torch.optim.AdamW)


class TestOptimizerRegistry:
    """Test suite for the optimizer registry pattern."""

    def test_registry_contains_default_optimizers(self):
        """Test that default optimizers are registered."""
        assert "adamw" in OPTIMIZER_REGISTRY
        assert "muon" in OPTIMIZER_REGISTRY

    def test_custom_optimizer_registration(self):
        """Test that custom optimizers can be registered."""
        
        @register_optimizer("test_custom")
        class GetOptimizerTestCustom:
            @staticmethod
            def create_optimizer(model_parameters, config):
                # Return a simple SGD optimizer for testing
                return torch.optim.SGD(model_parameters, lr=config.lr)
        
        # Verify registration
        assert "test_custom" in OPTIMIZER_REGISTRY
        
        # Test usage
        model = create_test_model()
        config = MockConfig(lr=0.01)
        
        optimizer = create_optimizer("test_custom", model.parameters(), config)
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.01
        
        # Clean up
        del OPTIMIZER_REGISTRY["test_custom"]

    def test_registry_error_provides_helpful_message(self):
        """Test that registry errors include available optimizers."""
        model = create_test_model()
        config = MockConfig()
        
        with pytest.raises(ValueError) as excinfo:
            create_optimizer("nonexistent", model.parameters(), config)
        
        error_message = str(excinfo.value)
        assert "Unknown optimizer type: nonexistent" in error_message
        assert "Available optimizers:" in error_message
        assert "Register custom optimizers using @register_optimizer decorator" in error_message

    def test_optimizer_class_interface(self):
        """Test that optimizer classes follow the expected interface."""
        from skyrl_train.distributed.utils import GetOptimizerAdamW, GetOptimizerMuon
        
        # Test that classes have the required static method
        assert hasattr(GetOptimizerAdamW, "create_optimizer")
        assert callable(GetOptimizerAdamW.create_optimizer)
        
        # Skip Muon test if not available
        try:
            from muon import MuonWithAuxAdam
            assert hasattr(GetOptimizerMuon, "create_optimizer")
            assert callable(GetOptimizerMuon.create_optimizer)
        except ImportError:
            pytest.skip("Muon optimizer not available")

    def test_registry_decorator_preserves_class(self):
        """Test that the register_optimizer decorator preserves the original class."""
        
        class OriginalClass:
            @staticmethod
            def create_optimizer(model_parameters, config):
                return torch.optim.SGD(model_parameters, lr=config.lr)
        
        # Apply decorator
        decorated_class = register_optimizer("test_preserve")(OriginalClass)
        
        # Verify the class is preserved
        assert decorated_class is OriginalClass
        assert "test_preserve" in OPTIMIZER_REGISTRY
        assert OPTIMIZER_REGISTRY["test_preserve"] is OriginalClass
        
        # Clean up
        del OPTIMIZER_REGISTRY["test_preserve"]