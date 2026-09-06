import unittest
import torch
import torch.nn.functional as F
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.loss import language_model_loss
from src.training.optimizer import create_optimizer
from src.training.step import train_step, validation_step
from src.model.gpt import GPT

class TestLanguageModelLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')

    def test_loss_shape(self):
        # 1. LOSS SHAPE
        logits = torch.randn(2, 4, 10, device=self.device)
        targets = torch.randint(0, 10, (2, 4), device=self.device)
        loss = language_model_loss(logits, targets)
        self.assertEqual(loss.dim(), 0)  # scalar

    def test_independent_cross_entropy_reference(self):
        # 2. INDEPENDENT CROSS-ENTROPY REFERENCE
        B, T, V = 2, 3, 5
        torch.manual_seed(99)
        logits = torch.randn(B, T, V, device=self.device)
        targets = torch.randint(0, V, (B, T), device=self.device)

        loss = language_model_loss(logits, targets)

        # Independent calculation
        logits_flat = logits.view(B * T, V)
        targets_flat = targets.view(B * T)

        # Manual cross entropy: -log(softmax(logits)[target])
        log_softmax = logits_flat - logits_flat.logsumexp(dim=-1, keepdim=True)
        expected_loss = -log_softmax[torch.arange(B * T), targets_flat].mean()

        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-5))

    def test_perfect_prediction(self):
        # 3. PERFECT PREDICTION
        B, T, V = 1, 2, 5
        logits = torch.full((B, T, V), -100.0)
        targets = torch.tensor([[0, 3]])
        logits[0, 0, 0] = 100.0
        logits[0, 1, 3] = 100.0

        loss = language_model_loss(logits, targets)
        self.assertLess(loss.item(), 0.01)

    def test_worst_prediction(self):
        # 4. WORST PREDICTION
        B, T, V = 1, 2, 5
        logits = torch.full((B, T, V), 100.0)
        targets = torch.tensor([[0, 3]])
        logits[0, 0, 0] = -100.0
        logits[0, 1, 3] = -100.0

        loss = language_model_loss(logits, targets)
        self.assertGreater(loss.item(), 10.0)

    def test_logit_shift_invariance(self):
        # 5. LOGIT SHIFT INVARIANCE
        B, T, V = 2, 3, 8
        logits = torch.randn(B, T, V, device=self.device)
        targets = torch.randint(0, V, (B, T), device=self.device)

        loss1 = language_model_loss(logits, targets)
        loss2 = language_model_loss(logits + 42.0, targets)

        self.assertTrue(torch.allclose(loss1, loss2, atol=1e-4))

    def test_gradient_flow(self):
        # 6. GRADIENT FLOW
        logits = torch.randn(2, 3, 10, requires_grad=True, device=self.device)
        targets = torch.randint(0, 10, (2, 3), device=self.device)

        loss = language_model_loss(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0))

    def test_numerical_validity(self):
        # 15. NUMERICAL VALIDITY
        logits = torch.randn(2, 4, 10, device=self.device)
        targets = torch.randint(0, 10, (2, 4), device=self.device)
        loss = language_model_loss(logits, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_edge_cases(self):
        # 14. EDGE CASES
        cases = [
            (1, 1, 2),
            (1, 1, 1),
            (2, 3, 5),
        ]
        for B, T, V in cases:
            logits = torch.randn(B, T, V, device=self.device)
            targets = torch.randint(0, V, (B, T), device=self.device)
            loss = language_model_loss(logits, targets)
            self.assertEqual(loss.dim(), 0)


class TestOptimizer(unittest.TestCase):
    def test_optimizer_creation(self):
        # 7. OPTIMIZER CREATION
        torch.manual_seed(42)
        model = GPT(8, 16, 8, 2, 32, 1)
        optimizer = create_optimizer(model, learning_rate=1e-3)

        self.assertIsInstance(optimizer, torch.optim.AdamW)

        # Verify all model parameters are registered
        model_params = set(id(p) for p in model.parameters())
        optim_params = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                optim_params.add(id(p))

        self.assertEqual(model_params, optim_params)


class TestTrainStep(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.model = GPT(8, 16, 8, 2, 32, 1).to(self.device)
        self.optimizer = create_optimizer(self.model, learning_rate=1e-2)

    def test_training_step_reduces_loss(self):
        # 8. TRAINING STEP REDUCES LOSS
        torch.manual_seed(42)
        model = GPT(8, 16, 8, 2, 32, 1).to(self.device)
        optimizer = create_optimizer(model, learning_rate=1e-2)

        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        initial_loss = train_step(model, optimizer, input_ids, targets).item()

        for _ in range(50):
            loss = train_step(model, optimizer, input_ids, targets)

        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss * 0.5)

    def test_parameters_actually_change(self):
        # 9. PARAMETERS ACTUALLY CHANGE
        param_before = self.model.lm_head.linear.weight.clone()

        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        train_step(self.model, self.optimizer, input_ids, targets)

        param_after = self.model.lm_head.linear.weight
        self.assertFalse(torch.equal(param_before, param_after))

    def test_train_mode(self):
        # 12. TRAIN/EVAL MODE
        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        train_step(self.model, self.optimizer, input_ids, targets)
        self.assertTrue(self.model.training)

    def test_train_step_returns_scalar(self):
        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        loss = train_step(self.model, self.optimizer, input_ids, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


class TestValidationStep(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.model = GPT(8, 16, 8, 2, 32, 1).to(self.device)

    def test_validation_does_not_update_parameters(self):
        # 10. VALIDATION DOES NOT UPDATE PARAMETERS
        params_before = {n: p.clone() for n, p in self.model.named_parameters()}

        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        validation_step(self.model, input_ids, targets)

        for name, param in self.model.named_parameters():
            self.assertTrue(torch.equal(param, params_before[name]),
                            f"Parameter {name} changed during validation")

    def test_validation_has_no_gradients(self):
        # 11. VALIDATION HAS NO GRADIENTS
        self.model.zero_grad()

        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        validation_step(self.model, input_ids, targets)

        for name, param in self.model.named_parameters():
            self.assertTrue(param.grad is None or torch.all(param.grad == 0),
                            f"Gradient found for {name} after validation")

    def test_eval_mode(self):
        # 12. TRAIN/EVAL MODE
        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        validation_step(self.model, input_ids, targets)
        self.assertFalse(self.model.training)

    def test_validation_returns_scalar(self):
        input_ids = torch.tensor([[0, 1, 2, 3]], device=self.device)
        targets = torch.tensor([[1, 2, 3, 4]], device=self.device)

        loss = validation_step(self.model, input_ids, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


class TestDeviceTraining(unittest.TestCase):
    def test_device_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available.")
        torch.manual_seed(42)
        model = GPT(8, 16, 8, 2, 32, 1).to('cuda')
        optimizer = create_optimizer(model, 1e-3)
        ids = torch.tensor([[0, 1, 2, 3]], device='cuda')
        tgt = torch.tensor([[1, 2, 3, 4]], device='cuda')

        loss = train_step(model, optimizer, ids, tgt)
        self.assertEqual(loss.device.type, 'cuda')

        val_loss = validation_step(model, ids, tgt)
        self.assertEqual(val_loss.device.type, 'cuda')

if __name__ == "__main__":
    unittest.main()
