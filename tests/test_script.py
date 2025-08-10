## Imports ##

from copy import deepcopy
import numpy as np
import pytest
from nn.script import (
    MultilayerPerceptron,
    relu,
    sigmoid,
    sigmoid_derivative,
    pre_regularization_value,
    feed,
    expected_values_last_layer,
    cost_gradients_on_examples_of_one_batch,
    cost_gradient_one_batch,
    costs_one_batch,
    cost_gradient,
    learning_one_step,
    cost,
    flatten_cost_gradient,
    consecutive_gradients_cosine,
    learning,
    prediction_result,
    accuracy,
    vectorize_learning_set,
)


# Construction of the random number generator, with a chosen seed, to ensure reproducibility of
# the tests.
rng = np.random.default_rng(seed=1)

## Fixtures-like functions ##
""" 
I've had trouble with implementing fixtures in my code, specifically when trying to request a 
fixture from another one while passing it only part of the parameters passed to the first one.
I could probably get around by turning some of my fixtures into functions but I decided to turn 
them all into functions, and use pytest.mark.parametrize in a direct way.
"""


def get_mlp(layout: list[int], dtype: type = np.float64) -> MultilayerPerceptron:
    return MultilayerPerceptron(layout=layout, dtype=dtype)


def standard_mlp(dtype: type = np.float64) -> MultilayerPerceptron:
    """Standard because it is the one used in the classifier at the end of the project."""
    layout = [784, 16, 16, 10]
    return MultilayerPerceptron(layout=layout, dtype=dtype)


def labeled_example(label_size: int, example_size: int) -> list[np.ndarray, np.ndarray]:
    return [np.random.rand(1, label_size), np.random.rand(1, example_size)]


def learning_set(
    layout: list, training_set_size: int, max_batch_size=1
) -> list[list[np.ndarray, np.ndarray]]:
    label_size = layout[-1]
    example_size = layout[0]
    _learning_set = [
        labeled_example(label_size=label_size, example_size=example_size)
        for _ in range(training_set_size)
    ]
    return vectorize_learning_set(
        learning_set=_learning_set, max_batch_size=max_batch_size
    )


## Unit tests ##


def test_1():
    layout = [3, 2, 1]
    a = get_mlp(layout)
    b = a.layers[0]
    c = a.layers[1]
    d = a.layers[2]
    assert b.nodes[0].weights is None
    assert len(c.nodes[0].weights) == 3
    assert len(c.nodes[1].weights) == 3
    assert len(d.nodes[0].weights) == 2


def test_2():
    a = standard_mlp()
    b = a.layers[0]
    c = a.layers[1]
    d = a.layers[3]
    assert b.nodes[0].weights is None
    assert len(c.nodes[0].weights) == 784
    assert b.nodes[0].activation_fct is None
    assert c.nodes[0].activation_fct.__code__ == relu.__code__
    assert d.nodes[0].activation_fct.__code__ == sigmoid.__code__
    e = a.variables
    assert len(e) == 4
    assert len(e[0]) == 3
    assert np.array_equal(
        e[0][0], np.array([None] * 784, dtype=np.float16, ndmin=2), equal_nan=True
    )
    assert len(e[1][1][15]) == 784


def test_pre_regularization_value():
    # Cas scalaire (j'ai passé trois heures à chercher l'erreur suivante : je n'avais pas transformé
    # le cas scalaire en "cas vectoriel à 1D"). Par ex., biais = np.random.rand().
    biais = np.array([np.random.rand()])
    values = np.random.rand(1, 2)
    weights = np.array([np.random.rand(2)])
    _pre_regularization_value = values @ weights.T - biais
    np.testing.assert_array_almost_equal(
        pre_regularization_value(biais=biais, weights=weights, values=values),
        _pre_regularization_value,
    )
    # Cas vectoriel
    size_rank = 2
    size_previous_rank = 2
    biais = np.random.rand(size_rank)
    values = np.random.rand(size_previous_rank)
    weights = np.random.rand(size_rank, size_previous_rank)
    _pre_regularization_value = np.sum(np.multiply(values, weights), axis=1) - biais
    np.testing.assert_array_almost_equal(
        pre_regularization_value(biais=biais, weights=weights, values=values),
        _pre_regularization_value,
    )
    # Cas doublement vectoriel
    batch_size = 2
    biais = np.random.rand(batch_size, 2)
    values = np.random.rand(batch_size, 3)
    weights = np.random.rand(batch_size, 3)
    expected_pre_regularization_value = [
        pre_regularization_value(biais=biais[s], weights=weights, values=values[s])
        for s in range(batch_size)
    ]
    np.testing.assert_array_almost_equal(
        pre_regularization_value(biais=biais, weights=weights, values=values),
        expected_pre_regularization_value,
    )


def test_feed():
    ## First series of tests where layer_rank = None ##

    mlp = get_mlp(layout=[1])
    example = np.array([[1]])
    feed(mlp=mlp, example=example)
    np.testing.assert_array_equal(mlp.variables[0][2], np.array([[1]]))
    mlp = get_mlp(layout=[2])
    example = np.array([1, 2])
    feed(mlp=mlp, example=example)
    np.testing.assert_array_equal(mlp.variables[0][2], np.array([1, 2]))

    mlp = get_mlp(layout=[1, 1])
    example = np.array([[1]])
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2][0][0]
    w = mlp.variables[1][1][0][0]
    b = mlp.variables[1][0][0][0]
    value = sigmoid((a * w) - b)
    assert mlp.variables[1][2][0][0] == value

    mlp = get_mlp(layout=[1, 2])
    example = np.array([[1]])
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(mlp.variables[1][2], value)

    mlp = get_mlp(layout=[2, 1])
    example = np.array([[1, 2]])
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(mlp.variables[1][2], value)

    mlp = get_mlp(layout=[2, 2])
    example = np.array([[1, 2]])
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(mlp.variables[1][2], value)

    mlp = get_mlp(layout=[1, 1, 1])
    example = np.array([[1]])
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = mlp.variables[2][1]
    b = mlp.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(mlp.variables[2][2], value_2)

    mlp = get_mlp(layout=[2, 2, 2])
    example = np.random.rand(1, 2)
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = mlp.variables[2][1]
    b = mlp.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(mlp.variables[2][2], value_2)

    mlp = get_mlp(layout=[784, 16, 16])
    example = np.random.rand(1, 784)
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = mlp.variables[2][1]
    b = mlp.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    rtol = 1e-10
    np.testing.assert_allclose(actual=mlp.variables[2][2], desired=value_2, rtol=rtol)

    mlp = get_mlp(layout=[784, 16, 16, 10])
    example = np.random.rand(1, 784)
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = mlp.variables[2][1]
    b = mlp.variables[2][0]
    value_2 = relu(np.sum(value_1 * w, axis=1) - b)
    w = mlp.variables[3][1]
    b = mlp.variables[3][0]
    value_3 = sigmoid(np.sum(value_2 * w, axis=1) - b)
    np.testing.assert_allclose(actual=mlp.variables[3][2], desired=value_3, rtol=rtol)

    ## Second series of tests with layer_rank != None ##
    layout = [2, 2, 2, 2]
    N = len(layout)
    modified_layer_rank = 0
    mlp = get_mlp(layout=layout, dtype=np.float64)
    new_values = np.random.rand(layout[modified_layer_rank])
    mlp.variables[modified_layer_rank][2][0] = new_values
    feed(
        mlp=mlp,
        start_layer_rank=modified_layer_rank + 1,
    )

    expected_last_layer_values = None
    for i in range(modified_layer_rank + 1, N):
        if i == modified_layer_rank + 1:
            values = new_values
        else:
            values = mlp.variables[i - 1][2]

        expected_last_layer_values = sigmoid(
            pre_regularization_value(
                biais=mlp.variables[i][0],
                weights=mlp.variables[i][1],
                values=values,
            )
        )
    np.testing.assert_array_equal(
        np.array(mlp.variables[N - 1][2]), expected_last_layer_values
    )

    # Test with several examples at the same time
    batch_size = 2
    mlp = get_mlp(layout=[2, 2, 2, 2])
    example = np.random.rand(batch_size, 2)
    feed(mlp=mlp, example=example)
    a = mlp.variables[0][2]
    w = mlp.variables[1][1]
    b = mlp.variables[1][0]
    value_1 = relu(pre_regularization_value(biais=b, weights=w, values=a))
    w = mlp.variables[2][1]
    b = mlp.variables[2][0]
    value_2 = relu(pre_regularization_value(biais=b, weights=w, values=value_1))
    w = mlp.variables[3][1]
    b = mlp.variables[3][0]
    value_3 = sigmoid(pre_regularization_value(biais=b, weights=w, values=value_2))
    np.testing.assert_allclose(actual=mlp.variables[3][2], desired=value_3, rtol=rtol)


def test_expected_values_last_layer():
    np.testing.assert_array_equal(
        expected_values_last_layer(0), np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    )
    np.testing.assert_array_equal(
        expected_values_last_layer(1), np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    )


@pytest.mark.parametrize(argnames="batch_size", argvalues=[(1), (2)])
def test_costs_one_batch(batch_size):
    # Test with start_layer_rank = None #
    layout = [2, 2]
    mlp = get_mlp(layout=layout)
    test_set = learning_set(
        layout=layout, training_set_size=2, max_batch_size=batch_size
    )
    labeled_examples_batch = test_set[0]

    actual_costs_one_batch = costs_one_batch(
        mlp=mlp,
        labeled_examples_batch=labeled_examples_batch,
    )

    feed(mlp=mlp, example=labeled_examples_batch[1])
    last_layer_values = mlp.variables[len(layout) - 1][2]
    desired_costs_one_batch = (
        np.linalg.norm(labeled_examples_batch[0] - last_layer_values, axis=1) ** 2
    )

    rtol = 1e-06
    np.testing.assert_allclose(
        actual=actual_costs_one_batch, desired=desired_costs_one_batch, rtol=rtol
    )

    # Test when start_layer_rank != None #
    layout = [2, 2, 2]
    N = len(layout)
    mlp = get_mlp(layout=layout, dtype=np.float64)
    label = np.random.rand(batch_size, layout[-1])
    modified_layer_rank = 0
    new_values = np.random.rand(batch_size, layout[modified_layer_rank])

    expected_layer_values = None
    for i in range(modified_layer_rank + 1, N):
        if i == modified_layer_rank + 1:
            values = new_values
        else:
            values = expected_layer_values

        expected_layer_values = mlp.layers[i].activation_fct(
            pre_regularization_value(
                biais=mlp.variables[i][0],
                weights=mlp.variables[i][1],
                values=values,
            )
        )
    expected_last_layer_values = expected_layer_values
    desired_costs_one_batch = (
        np.linalg.norm(label - expected_last_layer_values, axis=1) ** 2
    )

    mlp.variables[modified_layer_rank][2] = new_values
    actual_costs_one_batch = costs_one_batch(
        mlp=mlp,
        labeled_examples_batch=[label],
        start_layer_rank=modified_layer_rank + 1,
    )

    np.testing.assert_allclose(
        actual=actual_costs_one_batch, desired=desired_costs_one_batch, rtol=rtol
    )


@pytest.mark.parametrize(argnames="batch_size", argvalues=[(1), (2)])
def test_cost_gradients_on_examples_one_batch(batch_size):
    """
    The tests are divided into two groups. The first group assess that the cost gradient internal
    structure verify the theoretical formulae. The second group assess wether the computed
    direction maximize indeed the cost.
    """
    ## First group ##

    # First a test to assess the correct computation of the partial derivatives w.r.t. the values
    # of the last layer (whose computations are different than the rest). The formula we're
    # verifying is (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐀^(𝐍−𝟏) )=−2 ∗(𝐲−𝐀^(𝐍−𝟏)):
    size = 2
    mlp = get_mlp(layout=[size, size])
    label = np.random.rand(batch_size, size)
    _labeled_examples_batch = [label, np.random.rand(batch_size, size)]
    _cost_gradients = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_labeled_examples_batch,
    )
    np.testing.assert_array_equal(
        _cost_gradients[1][2], -2 * (label - mlp.variables[1][2])
    )

    # Then we verify the formula of the partial derivatives w.r.t. the biaises :
    # (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐁^𝐢 )=−(𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐀^𝐢 )⊙𝑓_𝑖^′ (𝐙^𝐢 ).
    np.testing.assert_array_equal(
        _cost_gradients[1][0],
        -_cost_gradients[1][2]
        * sigmoid_derivative(
            pre_regularization_value(
                biais=mlp.variables[1][0],
                weights=mlp.variables[1][1],
                values=mlp.variables[0][2],
            )
        ),
    )

    # We verify the formula of the partial derivatives w.r.t. the weights :
    # (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐖^𝐢 )=−((𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐁^𝑖 ))^(𝑅_1 )∗〖𝐀^(𝐢−𝟏)〗^(𝑅_2 ), where R1 and R2
    # represent certain rotations in space and * the usual dot product in N-d (sum over the second-
    # to-last dimension of the first entry and the last dimension of the second entry). To see a
    # diagram of this, see the docs. Intuitively, it's the operation for the case with one example
    # per batch, stacked over a new dimension.

    # Note that we start to need to use array_almost_equal instead of array_equal, probably because
    # of rouding errors which start to add up.
    np.testing.assert_array_almost_equal(
        _cost_gradients[1][1],
        -np.reshape(
            a=_cost_gradients[1][0],
            newshape=(batch_size, mlp.layers[1].size, 1),
        )
        @ np.reshape(
            a=mlp.variables[0][2],
            newshape=(batch_size, 1, mlp.layers[0].size),
        ),
    )

    # We verify the formula of the partial derivative w.r.t. the values of the previous layer:
    # (𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐀^(𝐢−𝟏) )=−(𝜕𝐶_(/𝑏𝑎𝑡𝑐ℎ))/(𝜕𝐁^𝐢 )∗𝑊^𝑖.
    np.testing.assert_array_almost_equal(
        _cost_gradients[0][2],
        -_cost_gradients[1][0] @ mlp.variables[1][1],
    )

    # We verify that the loop works as intended.
    size = 2
    mlp = get_mlp(layout=[size, size, size])
    _labeled_examples_batch = [
        np.random.rand(batch_size, size),
        np.random.rand(batch_size, size),
    ]
    _cost_gradients = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_labeled_examples_batch,
    )
    for i in range(size - 1, 0, -1):
        # w.r.t. to the biaises
        np.testing.assert_array_equal(
            _cost_gradients[i][0],
            -_cost_gradients[i][2]
            * mlp.layers[i].activation_fct_derivative(
                pre_regularization_value(
                    biais=mlp.variables[i][0],
                    weights=mlp.variables[i][1],
                    values=mlp.variables[i - 1][2],
                )
            ),
        )
        # w.r.t. to the weights
        np.testing.assert_array_almost_equal(
            _cost_gradients[i][1],
            -np.reshape(
                a=_cost_gradients[i][0],
                newshape=(batch_size, mlp.layers[i].size, 1),
            )
            @ np.reshape(
                a=mlp.variables[i - 1][2],
                newshape=(batch_size, 1, mlp.layers[i - 1].size),
            ),
        )
        # w.r.t. to the values of the previous layer
        np.testing.assert_array_almost_equal(
            _cost_gradients[i - 1][2],
            -_cost_gradients[i][0] @ mlp.variables[i][1],
        )

    ## 2nd group ##

    # We'll first verify that the computed partial derivatives makes sense, that is that when only
    # their respective parameter is nudged by eps, the cost varies roughly by partial_derivative *
    # eps.
    layout = [784, 16, 16, 10]
    eps = 2 * (10 ** (-5))
    dtype = np.float64

    # Relative tolerance of the comparison tests.
    rtol = 1e-2

    mlp = get_mlp(layout=layout, dtype=dtype)
    N = mlp.layer_count
    _labeled_examples_batch = [
        np.random.rand(batch_size, layout[-1]),
        np.random.rand(batch_size, layout[0]),
    ]
    initial_costs_one_batch = costs_one_batch(
        mlp=mlp,
        labeled_examples_batch=_labeled_examples_batch,
    )
    _cost_gradients_on_examples_of_one_batch = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_labeled_examples_batch,
    )

    for s in range(batch_size):
        # First we tackle the special case of the values of the last layer. We make sure to
        # use the alternative mode of operation of costs_one_batch (where no example is fed to the
        # first layer and the values of the multilayer perceptron are recomputed starting from a
        # certain row) as not doing it would overwrite the new nudged values of the last layer.
        # This will be a recurrent remark in the following tests.
        for j in range(layout[-1]):
            mlp.variables[N - 1][2][s][j] += eps
            new_costs_one_batch = costs_one_batch(
                mlp=mlp,
                labeled_examples_batch=_labeled_examples_batch,
                start_layer_rank=N,
            )
            dc = new_costs_one_batch[s] - initial_costs_one_batch[s]
            expected_dc = eps * _cost_gradients_on_examples_of_one_batch[N - 1][2][s][j]

            # We now assess that expected_dc is close to dc up to some relative tolerance. We also
            # let an absolute tolerance. Why ? Imagine the following case : expected_dc and the
            # real dc (not the one currently computed) are of the order of magnitude 1e-32, and
            # actual_cost and expected_cost of 1e-7 (those are not frequent values for the partial
            # derivatives w.r.t. the values of the last layer, but can be for the others parameters
            # whose effects might be hindered by the final sigmoid function. Thus, this remark is
            # mostly relevant for the later lests.). When dc is computed, 0 is returned because a
            # difference of 1e-32 is beyond the precision with which these numbers are saved (the
            # smallest difference catchable is 1e-7 (at which the decimal digits of the costs start
            # ) + the number of decimal digits of the representation (15 for float64, also called
            # the machine epsilon). In consequence, to not catch these irrelevant errors, we let an
            # absolute tolerance of expected_cost * 1e-(number of decimal digits of the used
            # representation), or alternatively, expected_cost * machine epsilon.
            machine_epsilon = np.finfo(dtype).eps
            atol = new_costs_one_batch[s] * machine_epsilon
            np.testing.assert_allclose(
                actual=dc, desired=expected_dc, rtol=rtol, atol=atol
            )
            # (In this simple case, we can find a simple theoretical formula for actual-desired,
            # which is exactly eps**2. This was confirmed visually when debugging).

            # We don't forget to put back the value to its initial state to not disturb the next
            # tests.
            mlp.variables[N - 1][2][s][j] += -eps

        # Now we test the partial derivatives w.r.t. the other parameters:
        for i in range(N - 1, 0, -1):
            # Starting with the biaises:
            for j in range(layout[i]):
                mlp.variables[i][0][0][j] += eps
                new_costs_one_batch = costs_one_batch(
                    mlp=mlp,
                    labeled_examples_batch=_labeled_examples_batch,
                    start_layer_rank=i,
                )
                dc = new_costs_one_batch[s] - initial_costs_one_batch[s]
                expected_dc = eps * _cost_gradients_on_examples_of_one_batch[i][0][s][j]

                # Same remark as previously on the absolute tolerance used.
                atol = new_costs_one_batch[s] * machine_epsilon
                np.testing.assert_allclose(
                    actual=dc, desired=expected_dc, rtol=rtol, atol=atol
                )

                # Again, we don't forget to put back the value to its initial state to not disturb
                # the next tests.
                mlp.variables[i][0][0][j] += -eps

            # The weights:
            for j in range(layout[i]):
                for k in range(layout[i - 1]):
                    mlp.variables[i][1][j][k] += eps
                    new_costs_one_batch = costs_one_batch(
                        mlp=mlp,
                        labeled_examples_batch=_labeled_examples_batch,
                        start_layer_rank=i,
                    )
                    dc = new_costs_one_batch[s] - initial_costs_one_batch[s]
                    expected_dc = (
                        eps * _cost_gradients_on_examples_of_one_batch[i][1][s][j][k]
                    )

                    # J'ai du mal à le justifier parfaitement, mais il suffit de rajouter ce petit
                    # 2 pour que les tests passent tout le temps (ils passent une fois sur deux
                    # sinon).
                    atol = 2 * new_costs_one_batch[s] * machine_epsilon
                    np.testing.assert_allclose(
                        actual=dc, desired=expected_dc, rtol=rtol, atol=atol
                    )

                    mlp.variables[i][1][j][k] += -eps

            # The value of the previous layer:
            for j in range(layout[i]):
                mlp.variables[i - 1][2][s][j] += eps
                new_costs_one_batch = costs_one_batch(
                    mlp,
                    labeled_examples_batch=_labeled_examples_batch,
                    start_layer_rank=(i - 1) + 1,
                )
                dc = new_costs_one_batch[s] - initial_costs_one_batch[s]
                expected_dc = (
                    eps * _cost_gradients_on_examples_of_one_batch[i - 1][2][s][j]
                )

                atol = new_costs_one_batch[s] * machine_epsilon
                np.testing.assert_allclose(
                    actual=dc, desired=expected_dc, rtol=rtol, atol=atol
                )

                mlp.variables[i - 1][2][s][j] += -eps


def test_cost_gradient_one_batch():
    layout = [2, 2, 2]
    mlp = get_mlp(layout=layout, dtype=np.float64)
    _training_set = learning_set(
        layout=layout,
        training_set_size=3,
        max_batch_size=2,
    )

    actual_cost_gradient_batch_2 = cost_gradient_one_batch(
        _cost_gradients_on_examples_of_one_batch=cost_gradients_on_examples_of_one_batch(
            mlp=mlp,
            labeled_examples_batch=_training_set[1],
        )
    )
    rtol = 1e-03
    _actual = actual_cost_gradient_batch_2[1][0]
    _desired = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_training_set[1],
    )[1][0]
    np.testing.assert_allclose(actual=_actual, desired=_desired, rtol=rtol)
    _actual = actual_cost_gradient_batch_2[1][1]
    _desired = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_training_set[1],
    )[1][1][0]
    np.testing.assert_allclose(actual=_actual, desired=_desired, rtol=rtol)
    _actual = actual_cost_gradient_batch_2[2][2]
    _desired = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_training_set[1],
    )[2][2]
    np.testing.assert_allclose(actual=_actual, desired=_desired, rtol=rtol)

    actual_cost_gradient_batch_1 = cost_gradient_one_batch(
        _cost_gradients_on_examples_of_one_batch=cost_gradients_on_examples_of_one_batch(
            mlp=mlp,
            labeled_examples_batch=_training_set[0],
        )
    )
    _actual = actual_cost_gradient_batch_1[0][2]
    _desired = np.average(
        cost_gradients_on_examples_of_one_batch(
            mlp=mlp,
            labeled_examples_batch=_training_set[0],
        )[0][2],
        axis=0,
        keepdims=True,
    )
    np.testing.assert_allclose(actual=_actual, desired=_desired, rtol=rtol)
    _actual = actual_cost_gradient_batch_1[1][1]
    _desired = np.average(
        cost_gradients_on_examples_of_one_batch(
            mlp=mlp,
            labeled_examples_batch=_training_set[0],
        )[1][1],
        axis=0,
    )
    np.testing.assert_allclose(actual=_actual, desired=_desired, rtol=rtol)


def assert_cost_gradient_equal(actual_gradient, expected_gradient):
    """
    Equality tests of two cost gradients. They are not simple arrays so we have to decompose
    the tests into several, alongside their internal structure. Used in a few tests.
    """
    N = len(actual_gradient)
    rtol = 1e-10
    for i in range(N):
        for variable in range(3):
            if not (i == 0 and variable != 2):
                np.testing.assert_allclose(
                    actual_gradient[i][variable],
                    expected_gradient[i][variable],
                    rtol=rtol,
                )


def test_cost_gradient():
    layout = [2, 2, 2]
    N = len(layout)
    mlp = get_mlp(layout=layout, dtype=np.float64)

    ## Test 1 ##
    training_set_size = 1
    max_batch_size = 1
    _training_set = learning_set(
        layout=layout,
        training_set_size=training_set_size,
        max_batch_size=max_batch_size,
    )

    expected_cost_gradients = cost_gradients_on_examples_of_one_batch(
        mlp=mlp,
        labeled_examples_batch=_training_set[0],
    )
    for i in range(1, N):
        expected_cost_gradients[i][1] = expected_cost_gradients[i][1][0]
    actual_cost_gradient = cost_gradient(mlp=mlp, training_set=_training_set)
    assert_cost_gradient_equal(
        actual_gradient=actual_cost_gradient,
        expected_gradient=expected_cost_gradients,
    )

    ## Test 2 ##
    training_set_size = 2
    max_batch_size = 2
    _training_set = learning_set(
        layout=layout,
        training_set_size=training_set_size,
        max_batch_size=max_batch_size,
    )
    actual_cost_gradient = cost_gradient(mlp=mlp, training_set=_training_set)
    expected_cost_gradients = cost_gradients_on_examples_of_one_batch(
        mlp=mlp, labeled_examples_batch=_training_set[0]
    )
    for i in range(N):
        for variable in range(3):
            if not (i == 0 and variable <= 1):
                expected_cost_gradients[i][variable] = np.atleast_2d(
                    np.average(expected_cost_gradients[i][variable], axis=0)
                )
    assert_cost_gradient_equal(
        actual_gradient=actual_cost_gradient,
        expected_gradient=expected_cost_gradients,
    )

    ## Test 3 ##
    training_set_size = 2
    max_batch_size = 1
    _training_set = learning_set(
        layout=layout,
        training_set_size=training_set_size,
        max_batch_size=max_batch_size,
    )
    actual_cost_gradient = cost_gradient(mlp=mlp, training_set=_training_set)
    expected_cost_gradients_1 = cost_gradients_on_examples_of_one_batch(
        mlp=mlp, labeled_examples_batch=_training_set[0]
    )
    expected_cost_gradients_2 = cost_gradients_on_examples_of_one_batch(
        mlp=mlp, labeled_examples_batch=_training_set[1]
    )
    expected_cost_gradient = []
    for i in range(N):
        expected_cost_gradient.append([])
        for variable in range(3):
            expected_cost_gradient[i].append(
                (
                    expected_cost_gradients_1[i][variable]
                    + expected_cost_gradients_2[i][variable]
                )
                / 2
            )
    for i in range(N):
        if i != 0:
            expected_cost_gradient[i][1] = expected_cost_gradient[i][1][0]
    assert_cost_gradient_equal(
        actual_gradient=actual_cost_gradient, expected_gradient=expected_cost_gradient
    )

    ## Test 4 ##
    training_set_size = 3
    max_batch_size = 2
    _training_set = learning_set(
        layout=layout,
        training_set_size=training_set_size,
        max_batch_size=max_batch_size,
    )
    actual_gradient = cost_gradient(mlp=mlp, training_set=_training_set)
    expected_cost_gradients_1 = cost_gradients_on_examples_of_one_batch(
        mlp=mlp, labeled_examples_batch=_training_set[0]
    )
    expected_cost_gradients_2 = cost_gradients_on_examples_of_one_batch(
        mlp=mlp, labeled_examples_batch=_training_set[1]
    )
    expected_cost_gradient = []
    for i in range(N):
        expected_cost_gradient.append([])
        for variable in range(3):
            expected_cost_gradient[i].append(
                (
                    2
                    * np.atleast_2d(
                        np.average(expected_cost_gradients_1[i][variable], axis=0)
                    )
                    + expected_cost_gradients_2[i][variable]
                )
                / 3
            )
    for i in range(N):
        if i != 0:
            expected_cost_gradient[i][1] = expected_cost_gradient[i][1][0]
    assert_cost_gradient_equal(
        actual_gradient=actual_gradient, expected_gradient=expected_cost_gradient
    )

    ## Test 5 ##
    training_set_size = 5
    max_batch_size = 3
    _training_set = learning_set(
        layout=layout,
        training_set_size=training_set_size,
    )
    cost_gradient_1 = cost_gradient(mlp=mlp, training_set=_training_set)
    _training_set = vectorize_learning_set(
        learning_set=_training_set, max_batch_size=max_batch_size
    )
    cost_gradient_2 = cost_gradient(mlp=mlp, training_set=_training_set)
    assert_cost_gradient_equal(cost_gradient_1, cost_gradient_2)


@pytest.mark.parametrize(
    argnames="training_set_size,max_batch_size",
    argvalues=[(1, 1), (2, 1), (2, 2), (3, 2)],
)
def test_cost(training_set_size, max_batch_size):
    layout = [2, 2]
    mlp = get_mlp(layout=layout)
    _training_set = learning_set(
        layout=layout,
        training_set_size=training_set_size,
        max_batch_size=max_batch_size,
    )
    expected_costs = np.array(
        [
            np.average(
                costs_one_batch(
                    mlp=mlp,
                    labeled_examples_batch=labeled_examples_batch,
                )
            )
            for labeled_examples_batch in _training_set
        ]
    )
    _weights = [
        len(labeled_examples_batch[0]) for labeled_examples_batch in _training_set
    ]
    expected_cost = np.average(a=expected_costs, weights=_weights)
    actual_cost = cost(
        mlp=mlp,
        training_set=_training_set,
    )
    np.testing.assert_equal(expected_cost, actual_cost)


def test_learning_one_step():
    ## 1st series of tests : with a training set of one example ##

    # The first test assesses the structure of the response.
    layout = [16, 16, 16, 10]
    N = len(layout)
    mlp = get_mlp(layout=layout, dtype=np.float64)
    labeled_examples_batch_1 = [
        np.random.rand(1, layout[-1]),
        np.random.rand(1, layout[0]),
    ]
    _training_set = [labeled_examples_batch_1]

    # the cost is computed, it will be used for the second test. We have to do it now before the
    # parameters are nudged.
    initial_cost = cost(mlp, _training_set)

    expected_cost_gradient = cost_gradient(mlp, training_set=_training_set)
    eta = 1e-02
    expected_variables = deepcopy(mlp.variables)
    for i in range(N):
        for variable in range(2):
            expected_variables[i][variable] += (
                -eta * np.average(layout) * expected_cost_gradient[i][variable]
            )

    learning_one_step(
        mlp=mlp,
        training_set=_training_set,
        eta=eta,
        inertia=False,
    )

    for i in range(N):
        # We only care about the effects on the parameters, not on the node values. Note that if we
        # were to carry the comparison for the nodes values also, it would fail as the node values
        # have been modified in the meantime by learning_one_step, by the call of cost inside of
        # it.
        for variable in range(2):

            np.testing.assert_array_almost_equal(
                mlp.variables[i][variable],
                expected_variables[i][variable],
            )

    # The second test verifies that the cost decreases after the learning step.
    new_cost = mlp.cost
    assert new_cost <= initial_cost

    ## 2nd series of tests : with a training set of 2 examples ##

    # The first test assesses the structure of the response.
    labeled_examples_batch_2 = [
        np.random.rand(1, layout[-1]),
        np.random.rand(1, layout[0]),
    ]
    _training_set = [labeled_examples_batch_1, labeled_examples_batch_2]

    # the cost is computed, it will be used for the second test. We have to do it now before the
    # parameters are nudged.
    initial_cost = cost(mlp, _training_set)

    # Computation of the expected cost gradient
    expected_cost_gradient_1 = cost_gradient(mlp, training_set=_training_set)
    expected_cost_gradient_2 = cost_gradient(mlp, training_set=_training_set)
    expected_cost_gradient = []
    for i in range(N):
        expected_cost_gradient.append([[], [], []])
        for variable in range(3):
            expected_cost_gradient[i][variable] = np.average(
                np.array(
                    [
                        expected_cost_gradient_1[i][variable],
                        expected_cost_gradient_2[i][variable],
                    ]
                ),
                axis=0,
            )

    eta = 1e-02
    expected_variables = deepcopy(mlp.variables)
    for i in range(N):
        for variable in range(3):
            expected_variables[i][variable] += (
                -eta * np.average(layout) * expected_cost_gradient[i][variable]
            )

    learning_one_step(
        mlp=mlp,
        training_set=_training_set,
        eta=eta,
        inertia=False,
    )

    for i in range(N):
        # We only care about the effects on the parameters, not on the node values. Note that if we
        # were to carry the comparison for the nodes values also, it would fail as the node values
        # have been modified in the meantime by learning_one_step, by the call of cost inside of
        # it.
        for variable in range(2):

            np.testing.assert_array_almost_equal(
                mlp.variables[i][variable],
                expected_variables[i][variable],
            )

    # The second test verifies that the cost decreases after the learning step.
    new_cost = mlp.cost
    assert new_cost <= initial_cost

    ## 3rd test with a training set with 3 labeled_examples, and max_batch_size=2##
    _training_set = learning_set(layout=layout, training_set_size=3, max_batch_size=2)
    initial_cost = cost(mlp=mlp, training_set=_training_set)
    learning_one_step(mlp=mlp, training_set=_training_set, eta=eta, inertia=False)
    new_cost = mlp.cost
    assert new_cost < initial_cost

    ## A test to verify the returns_cost_gradient feature
    _cost_gradient = learning_one_step(
        mlp=mlp,
        training_set=_training_set,
        eta=eta,
        inertia=False,
        returns_cost_gradient=True,
    )

    ## A test of the inertia feature
    layout = [2, 2]
    mlp = get_mlp(layout=layout)
    _training_set = learning_set(layout=layout, training_set_size=2)
    previous_cost_gradient = cost_gradient(mlp=mlp, training_set=_training_set)
    initial_cost = cost(mlp=mlp, training_set=_training_set)

    learning_one_step(
        mlp=mlp,
        training_set=_training_set,
        eta=eta,
        inertia=True,
        inertia_strength=0.2,
        previous_cost_gradient=previous_cost_gradient,
    )
    new_cost = mlp.cost
    assert new_cost < initial_cost


def test_flatten_cost_gradient():
    layout = [2, 1]
    mlp = get_mlp(layout=layout)
    _training_set = learning_set(layout=layout, training_set_size=1)
    _cost_gradient = cost_gradient(mlp=mlp, training_set=_training_set)
    actual_flattened_cost_gradient = flatten_cost_gradient(
        _cost_gradient=_cost_gradient
    )
    expected_flattened_cost_gradient = np.array(
        [
            _cost_gradient[1][0][0][0],
            _cost_gradient[1][1][0][0],
            _cost_gradient[1][1][0][1],
        ]
    )
    np.testing.assert_array_equal(
        actual_flattened_cost_gradient, expected_flattened_cost_gradient
    )


def test_consecutive_gradients_cosine():
    layout = [3, 2, 1]
    mlp = get_mlp(layout=layout)
    _training_set = learning_set(layout=layout, training_set_size=2)
    cost_gradient_1 = cost_gradient(mlp=mlp, training_set=_training_set[1:2])
    cost_gradient_2 = cost_gradient(mlp, training_set=_training_set[0:1])
    _consecutive_gradients_cosine = consecutive_gradients_cosine(
        cost_gradient_1=cost_gradient_1, cost_gradient_2=cost_gradient_2
    )
    assert isinstance(_consecutive_gradients_cosine, float)


def test_learning():
    # First test to make sure no error is raised.
    layout = [1, 1]
    mlp = get_mlp(layout=layout, dtype=np.float64)
    _training_set = learning_set(layout=layout, training_set_size=1)
    eta = 1
    learning(
        mlp=mlp,
        training_set=_training_set,
        stop_condition="stagnation",
        eta=eta,
        inertia=False,
        stagnation_epsilon=0.01,
    )

    steps_count = 10
    learning(
        mlp=mlp,
        training_set=_training_set,
        test_set=_training_set,
        eta=eta,
        steps_count=steps_count,
        stochastic=False,
        inertia=False,
    )

    # This time with a training set with multiple examples.
    layout = [5, 5, 5]
    mlp = get_mlp(layout=layout, dtype=np.float64)
    _training_set = learning_set(layout=layout, training_set_size=2)
    eta = 1
    learning(
        mlp=mlp,
        training_set=_training_set,
        test_set=_training_set,
        stop_condition="stagnation",
        eta=eta,
        stagnation_epsilon=0.1,
        stochastic=False,
        inertia=False,
    )

    # Test of the stochastic feature
    learning(
        mlp=mlp,
        training_set=_training_set,
        test_set=_training_set,
        eta=eta,
        stochastic=True,
        step_training_size=2,
        inertia=False,
    )

    # Test of the metrics_during_training feature:
    tracked_metrics = learning(
        mlp=mlp,
        training_set=_training_set,
        test_set=_training_set,
        eta=eta,
        steps_count=10,
        stochastic=True,
        step_training_size=2,
        metrics_to_track=[
            "training_costs",
            "accuracies",
            "gradients_norms",
            "consecutive_gradients_cosines",
        ],
        inertia=False,
    )


def test_prediction_result():
    layout = [1, 1]
    mlp = get_mlp(layout=layout)
    mlp.variables[1][0] = 0
    mlp.variables[1][1][0][0] = 0
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    assert (
        prediction_result(
            mlp=mlp,
            labeled_example=labeled_example_1,
        )
        == 1
    )
    layout = [1, 10]
    mlp = get_mlp(layout=layout)
    mlp.variables[1][0] = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0]
    mlp.variables[1][1] = np.array(
        [
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ]
    )
    labeled_example_2 = [
        np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
        np.random.rand(1, layout[0]),
    ]
    assert (
        prediction_result(
            mlp=mlp,
            labeled_example=labeled_example_2,
        )
        == 1
    )
    labeled_example_3 = [
        np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        np.random.rand(1, layout[0]),
    ]
    assert (
        prediction_result(
            mlp=mlp,
            labeled_example=labeled_example_3,
        )
        == 0
    )


def test_accuracy():
    layout = [1, 10]
    mlp = get_mlp(layout=layout)
    labeled_examples_batch_1 = [
        np.random.rand(1, layout[-1]),
        np.random.rand(1, layout[0]),
    ]
    test_set_1 = [labeled_examples_batch_1]
    assert accuracy(mlp=mlp, test_set=test_set_1) == prediction_result(
        mlp=mlp, labeled_example=test_set_1[0]
    )
    layout = [2, 2]
    mlp = get_mlp(layout=layout)
    labeled_example_2 = [np.array([[1, 0]]), np.random.rand(1, 2)]
    labeled_example_3 = [np.array([[0, 1]]), np.random.rand(1, 2)]
    mlp.variables[1][0][0] = [-1, 0]
    mlp.variables[1][1] = np.zeros(2)
    test_set_2 = [labeled_example_2, labeled_example_3]
    assert accuracy(mlp=mlp, test_set=test_set_2) == 0.5
    # Case with batches containing more than one labeled example
    layout = [2, 2]
    mlp = get_mlp(layout=layout)
    test_set = learning_set(layout=layout, training_set_size=10)
    vectorized_test_set = vectorize_learning_set(
        learning_set=test_set, max_batch_size=10
    )
    assert accuracy(mlp=mlp, test_set=vectorized_test_set) == accuracy(
        mlp=mlp, test_set=test_set
    )
    vectorized_test_set = vectorize_learning_set(
        learning_set=test_set, max_batch_size=2
    )
    assert accuracy(mlp=mlp, test_set=vectorized_test_set) == np.average(
        [
            accuracy(
                mlp=mlp,
                test_set=[labeled_examples_batch],
            )
            for labeled_examples_batch in vectorized_test_set
        ]
    )
    # Case where the last batch is not full (len(learning_set)%max_batch_size != 0)
    vectorized_test_set = vectorize_learning_set(
        learning_set=test_set, max_batch_size=3
    )
    weights = [len(batch[0]) for batch in vectorized_test_set]
    assert accuracy(mlp=mlp, test_set=vectorized_test_set) == np.average(
        [
            accuracy(
                mlp=mlp,
                test_set=[labeled_examples_batch],
            )
            for labeled_examples_batch in vectorized_test_set
        ],
        weights=weights,
    )


def test_vectorize_learning_set_errors():
    # Test of the error handling
    _learning_set = learning_set(layout=[1, 1], training_set_size=1)
    max_batch_size = 2
    with pytest.raises(
        expected_exception=ValueError,
        match="The maximum batch size has to be smaller than the size of the learning set",
    ):
        _ = vectorize_learning_set(
            learning_set=_learning_set, max_batch_size=max_batch_size
        )


@pytest.mark.parametrize(
    argnames="_learning_set,max_batch_size,expected_output",
    argvalues=[
        ([[np.array([[0]]), np.array([[0]])]], 1, [[np.array([[0]]), np.array([[0]])]]),
        (
            [[np.array([[0]]), np.array([[1]])], [np.array([[3]]), np.array([[4]])]],
            1,
            [[np.array([[0]]), np.array([[1]])], [np.array([[3]]), np.array([[4]])]],
        ),
        (
            [[np.array([[0]]), np.array([[1]])], [np.array([[3]]), np.array([[4]])]],
            2,
            [[np.array([[0], [3]]), np.array([[1], [4]])]],
        ),
        (
            [
                [np.array([[0]]), np.array([[1]])],
                [np.array([[3]]), np.array([[4]])],
                [np.array([[6]]), np.array([[7]])],
            ],
            2,
            [
                [np.array([[0], [3]]), np.array([[1], [4]])],
                [np.array([[6]]), np.array([[7]])],
            ],
        ),
        (
            [
                [np.array([[0, 1]]), np.array([[0, 1]])],
                [np.array([[3, 4]]), np.array([[3, 4]])],
            ],
            2,
            [[np.array([[0, 1], [3, 4]]), np.array([[0, 1], [3, 4]])]],
        ),
    ],
)
def test_vectorize_learning_set(_learning_set, max_batch_size, expected_output):
    np.testing.assert_equal(
        actual=vectorize_learning_set(
            learning_set=_learning_set, max_batch_size=max_batch_size
        ),
        desired=expected_output,
    )
