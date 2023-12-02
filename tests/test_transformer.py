import pytest
import torch
from torch import nn

from customer_analysis.models.nn.transformer import TransformerModel


@pytest.mark.parametrize("input_size, model_size, num_layers,\
                         dim_feedforward, device, num_heads,\
                         dropout_rate, task", [
    (10, 20, 2, 40, 'cpu', 2, 0.05, 'events'),
    (15, 30, 3, 50, 'cpu', 3, 0.1, 'events'),
    (20, 40, 4, 60, 'cpu', 4, 0.15, 'events')])
def test_TransformerModel_init(input_size,
                               model_size,
                               num_layers,
                               dim_feedforward,
                               device,
                               num_heads,
                               dropout_rate,
                               task):
    """
    Tests the '__init__' method of the TransformerModel class.
    """
    model = TransformerModel(input_size=input_size,
                             model_size=model_size,
                             num_layers=num_layers,
                             dim_feedforward=dim_feedforward,
                             device=device,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate,
                             task=task)

    assert isinstance(model.embedding, nn.Embedding)
    assert isinstance(model.fc, nn.Linear)


@pytest.mark.parametrize("input_size, model_size, num_layers,\
                         dim_feedforward, device, num_heads,\
                         dropout_rate, task", [
    (10, 20, 2, 40, 'cpu', 2, .05, 'events'),
    (15, 30, 3, 50, 'cpu', 3, .1, 'events'),
    (20, 40, 4, 60, 'cpu', 4, .15, 'events')])
def test_TransformerModel_forward(input_size,
                                  model_size,
                                  num_layers,
                                  dim_feedforward,
                                  device,
                                  num_heads,
                                  dropout_rate,
                                  task):
    """
    Tests the 'forward' method of the TransformerModel class.
    """
    model = TransformerModel(input_size=input_size,
                             model_size=model_size,
                             num_layers=num_layers,
                             dim_feedforward=dim_feedforward,
                             device=device,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate,
                             task=task)

    seq = torch.randint(0, input_size, (3, max([4])))
    inputs_mask = torch.zeros(3, max([4]))

    out = model.forward(seq=seq.long(), inputs_mask=inputs_mask.bool())

    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], list)


@pytest.mark.parametrize("input_size, model_size, num_layers,\
                         dim_feedforward,device, num_heads,\
                         dropout_rate, sequence, inputs_mask", [
    (10, 20, 2, 40, 'cpu', 2, 0.05,
     torch.randint(0, 10, (4,)), torch.tensor([False]*4)),
    (15, 30, 3, 60, 'cpu', 3, 0.1,
     torch.randint(0, 15, (4,)), torch.tensor([False]*4)),
    (20, 40, 4, 80, 'cpu', 4, 0.15,
     torch.randint(0, 20, (4,)), torch.tensor([False]*4))])
def test_TransformerModel_predict(input_size,
                                  model_size,
                                  num_layers,
                                  dim_feedforward,
                                  device,
                                  num_heads,
                                  dropout_rate,
                                  sequence,
                                  inputs_mask):
    """
    Tests the 'predict' method of the TransformerModel class.
    """
    model = TransformerModel(input_size=input_size,
                             model_size=model_size,
                             num_layers=num_layers,
                             dim_feedforward=dim_feedforward,
                             device=device,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate)

    prediction, _ = model.predict(
        sequence.unsqueeze(0), inputs_mask.unsqueeze(0))

    assert prediction in range(input_size)


@pytest.mark.parametrize("input_size, model_size, num_layers,\
                         dim_feedforward, device, num_heads,\
                         dropout_rate, task, inputs, inputs_mask, targets", [
    (10, 20, 8, 40, 'cpu', 2, 0.05, 'events',
     torch.randint(0, 10, (3, 4)),
     torch.tensor([[False]*4]*3),
     torch.randint(0, 10, (3,))),
    (20, 40, 8, 40, 'cpu', 4, 0.1, 'events',
     torch.randint(0, 10, (4, 4)),
     torch.tensor([[False]*4]*4),
     torch.randint(0, 15, (4,)))])
def test_TransformerModel_step(input_size,
                               model_size,
                               num_layers,
                               dim_feedforward,
                               device,
                               num_heads,
                               dropout_rate,
                               task,
                               inputs,
                               inputs_mask,
                               targets):
    """
    Tests the 'step' method of the TransformerModel class.
    """
    model = TransformerModel(input_size=input_size,
                             model_size=model_size,
                             num_layers=num_layers,
                             dim_feedforward=dim_feedforward,
                             device=device,
                             num_heads=num_heads,
                             dropout_rate=dropout_rate,
                             task=task)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = model.step('train',
                      inputs=inputs.long(),
                      inputs_mask=inputs_mask.bool(),
                      targets=targets.long(),
                      loss_func=loss_func,
                      optimizer=optimizer)

    assert isinstance(loss[0], float)
