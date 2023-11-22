import pytest
import torch
from torch import nn

from customer_analysis.models.nn.rnn import RNNModel


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads", [
    (20, 2, 5, 'cpu', 999, 'relu', 'global', 1),
    (25, 3, 6, 'cpu', 999, 'tanh', 'self', 1),
    (30, 4, 7, 'cpu', 999, 'relu', 'multi-head', 2)])
def test_RNNModel_init(hidden_size,
                       num_layers,
                       num_classes,
                       device,
                       padding_value,
                       nonlinearity,
                       attention_type,
                       num_heads):
    """
    Tests the '__init__' method of the RNNModel class.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    assert model.hidden_size == hidden_size
    assert model.num_layers == num_layers
    assert model.device == device
    assert model.padding_value == padding_value
    assert model.attention_type == attention_type


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads, inputs, seq_lengths", [
    (20, 2, 5, 'cpu', 999, 'relu', 'global', 1,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2]),
    (25, 3, 6, 'cpu', 999, 'tanh', 'self', 1,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2]),
    (30, 4, 7, 'cpu', 999, 'relu', 'multi-head', 2,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2])])
def test_RNNModel_forward(hidden_size,
                          num_layers,
                          num_classes,
                          device,
                          padding_value,
                          nonlinearity,
                          attention_type,
                          num_heads,
                          inputs,
                          seq_lengths):
    """
    Tests the 'forward' method of the RNNModel class.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    out, _ = model.forward(inputs.to(device), seq_lengths)

    assert out.shape == (inputs.shape[0], num_classes)


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads, inputs, seq_lengths, k", [
    (20, 2, 5, 'cpu', 999, 'relu', 'global', 1,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2], 1),
    (25, 3, 6, 'cpu', 999, 'tanh', 'self', 1,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2], 1),
    (30, 4, 7, 'cpu', 999, 'relu', 'multi-head', 2,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2], 1)])
def test_RNNModel_predict(hidden_size,
                          num_layers,
                          num_classes,
                          device,
                          padding_value,
                          nonlinearity,
                          attention_type,
                          num_heads,
                          inputs,
                          seq_lengths,
                          k):
    """
    Tests the 'predict' method of the RNNModel class.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    predictions = model.predict(inputs.to(device), seq_lengths, k)

    assert predictions.shape == (inputs.shape[0], k)


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads, inputs, \
                         seq_lengths, targets", [
    (20, 2, 5, 'cpu', 999, 'relu', 'global', 1,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2],
     torch.randint(0, 5, (3,))),
    (25, 3, 6, 'cpu', 999, 'tanh', 'self', 1,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2],
     torch.randint(0, 6, (3,))),
    (30, 4, 7, 'cpu', 999, 'relu', 'multi-head', 2,
     torch.randn(3, max([4, 3, 2]), 1), [4, 3, 2],
     torch.randint(0, 7, (3,),))])
def test_RNNModel_step(hidden_size,
                       num_layers,
                       num_classes,
                       device,
                       padding_value,
                       nonlinearity,
                       attention_type,
                       num_heads,
                       inputs,
                       seq_lengths,
                       targets):
    """
    Tests the 'step' method of the RNNModel class.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = model.step('train',
                      inputs=inputs.to(device),
                      seq_lengths=seq_lengths,
                      targets=targets.to(device),
                      loss_func=loss_func,
                      optimizer=optimizer)

    assert isinstance(loss[0], float)


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads, out", [
    (20, 2, 5, 'cpu', 999, 'relu', 'self', 1, torch.randn(3, 4, 20)),
    (25, 3, 6, 'cpu', 999, 'tanh', 'self', 1, torch.randn(3, 4, 25)),
    (30, 4, 7, 'cpu', 999, 'relu', 'self', 1, torch.randn(3, 4, 30))])
def test_RNNModel_attention_self(hidden_size,
                                 num_layers,
                                 num_classes,
                                 device,
                                 padding_value,
                                 nonlinearity,
                                 attention_type,
                                 num_heads,
                                 out):
    """
    Tests the '_attention_self' method of the RNNModel class.
    Note: Tests private function,
        but it's necessary to ensure that this function is correct.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    # Create a mask for the 'out' tensor
    mask = torch.ones(out.shape, dtype=torch.bool)

    result, _ = model._attention_self(out, mask)

    assert result.shape == (out.shape[0], out.shape[2])


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads, out", [
    (20, 2, 5, 'cpu', 999, 'relu', 'global', 1, torch.randn(3, 4, 20)),
    (25, 3, 6, 'cpu', 999, 'tanh', 'global', 1, torch.randn(3, 4, 25)),
    (30, 4, 7, 'cpu', 999, 'relu', 'global', 1, torch.randn(3, 4, 30))])
def test_RNNModel_attention_global(hidden_size,
                                   num_layers,
                                   num_classes,
                                   device,
                                   padding_value,
                                   nonlinearity,
                                   attention_type,
                                   num_heads,
                                   out):
    """
    Tests the '_attention_global' method of the RNNModel class.
    Note: Tests private function,
        but it's necessary to ensure that this function is correct.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    # Create a mask for the 'out' tensor
    mask = torch.ones(out.shape, dtype=torch.bool)

    result, _ = model._attention_global(out.to(device), mask.to(device))

    assert result.shape == (out.shape[0], out.shape[2])


@pytest.mark.parametrize("hidden_size, num_layers, num_classes, \
                         device, padding_value, nonlinearity, \
                         attention_type, num_heads, out", [
    (20, 2, 5, 'cpu', 999, 'relu', 'multi-head', 2, torch.randn(3, 4, 20)),
    (25, 3, 6, 'cpu', 999, 'tanh', 'multi-head', 3, torch.randn(3, 4, 25)),
    (30, 4, 7, 'cpu', 999, 'relu', 'multi-head', 4, torch.randn(3, 4, 30))])
def test_RNNModel_attention_multi_head(hidden_size,
                                       num_layers,
                                       num_classes,
                                       device,
                                       padding_value,
                                       nonlinearity,
                                       attention_type,
                                       num_heads,
                                       out):
    """
    Tests the '_attention_multi_head' method of the RNNModel class.
    Note: Tests private function,
        but it's necessary to ensure that this function is correct.
    """
    model = RNNModel(hidden_size=hidden_size,
                     num_layers=num_layers,
                     num_classes=num_classes,
                     device=device,
                     padding_value=padding_value,
                     nonlinearity=nonlinearity,
                     attention_type=attention_type,
                     num_heads=num_heads)

    # Create a mask for the 'out' tensor
    mask = torch.ones(out.shape, dtype=torch.bool)

    result, _ = model._attention_multi_head(out.to(device), mask.to(device))

    assert result.shape == (out.shape[0], out.shape[2])
