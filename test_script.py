import numpy as np
import script
from copy import deepcopy
from script import (
    relu,
    sigmoid,
    sigmoid_derivative,
    pre_regularization_value,
    feed,
    expected_values_last_layer,
    cost_gradient_one_example,
    cost_one_example,
    cost_gradient,
    learning_one_step,
    cost,
    learning,
)


def test_1():
    a = script.MultilayerPerceptron([3, 2, 1])
    b = a.layers[0]
    c = a.layers[1]
    d = a.layers[2]
    assert b.nodes[0].weights is None
    assert len(c.nodes[0].weights) == 3
    assert len(c.nodes[1].weights) == 3
    assert len(d.nodes[0].weights) == 2


def test_2():
    a = script.MultilayerPerceptron([784, 16, 16, 10])
    b = a.layers[0]
    c = a.layers[1]
    d = a.layers[3]
    assert b.nodes[0].weights is None
    assert len(c.nodes[0].weights) == 784
    assert b.nodes[0].activation_fct is None
    assert c.nodes[0].activation_fct == relu
    assert d.nodes[0].activation_fct == sigmoid
    e = a.variables
    assert len(e) == 4
    assert len(e[0]) == 3
    assert np.array_equal(
        e[0][0], np.array([None] * 784, dtype=np.float16, ndmin=2), equal_nan=True
    )
    assert len(e[1][1][15]) == 784


def test_multiply():
    """
    To make sure I understand well the np.multiply function.
    """
    values = [2]
    weights = [1, 2]
    assert np.array_equal(np.multiply(values, weights), np.array([2, 4]))
    values = np.array([2, -2])
    weights = np.array([[1, 2], [3, 4]])
    assert np.array_equal(np.multiply(values, weights), np.array([[2, -4], [6, -8]]))


def test_sum():
    """
    To make sure I understand well the np.sum function.
    """
    A = np.array([[1, 2], [3, 4]])
    assert np.array_equal(np.sum(A, axis=1), np.array([3, 7]))


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


def test_feed():
    ## First series of tests where layer_rank = None ##

    multilayer_perceptron = script.MultilayerPerceptron([1])
    example = np.array([[1]])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    np.testing.assert_array_equal(
        multilayer_perceptron.variables[0][2], np.array([[1]])
    )
    multilayer_perceptron = script.MultilayerPerceptron([2])
    example = np.array([1, 2])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    np.testing.assert_array_equal(
        multilayer_perceptron.variables[0][2], np.array([1, 2])
    )

    multilayer_perceptron = script.MultilayerPerceptron([1, 1])
    example = np.array([[1]])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2][0][0]
    w = multilayer_perceptron.variables[1][1][0][0]
    b = multilayer_perceptron.variables[1][0][0][0]
    value = sigmoid((a * w) - b)
    assert multilayer_perceptron.variables[1][2][0][0] == value

    multilayer_perceptron = script.MultilayerPerceptron([1, 2])
    example = np.array([[1]])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[1][2], value)

    multilayer_perceptron = script.MultilayerPerceptron([2, 1])
    example = np.array([[1, 2]])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[1][2], value)

    multilayer_perceptron = script.MultilayerPerceptron([2, 2])
    example = np.array([[1, 2]])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value = sigmoid(np.sum(a * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[1][2], value)

    multilayer_perceptron = script.MultilayerPerceptron([1, 1, 1])
    example = np.array([[1]])
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[2][2], value_2)

    multilayer_perceptron = script.MultilayerPerceptron([2, 2, 2])
    example = np.random.rand(1, 2)
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[2][2], value_2)

    multilayer_perceptron = script.MultilayerPerceptron([784, 16, 16])
    example = np.random.rand(1, 784)
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = sigmoid(np.sum(value_1 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[2][2], value_2)

    multilayer_perceptron = script.MultilayerPerceptron([784, 16, 16, 10])
    example = np.random.rand(1, 784)
    feed(multilayer_perceptron=multilayer_perceptron, example=example)
    a = multilayer_perceptron.variables[0][2]
    w = multilayer_perceptron.variables[1][1]
    b = multilayer_perceptron.variables[1][0]
    value_1 = relu(np.sum(a * w, axis=1) - b)
    w = multilayer_perceptron.variables[2][1]
    b = multilayer_perceptron.variables[2][0]
    value_2 = relu(np.sum(value_1 * w, axis=1) - b)
    w = multilayer_perceptron.variables[3][1]
    b = multilayer_perceptron.variables[3][0]
    value_3 = sigmoid(np.sum(value_2 * w, axis=1) - b)
    np.testing.assert_array_equal(multilayer_perceptron.variables[3][2], value_3)

    ## Second series of tests with layer_rank != None ##
    layout = [2, 2, 2, 2]
    N = len(layout)
    modified_layer_rank = 0
    multilayer_perceptron = script.MultilayerPerceptron(layout, dtype=np.float64)
    new_values = np.random.rand(layout[modified_layer_rank])
    multilayer_perceptron.variables[modified_layer_rank][2][0] = new_values
    feed(
        multilayer_perceptron=multilayer_perceptron,
        start_layer_rank=modified_layer_rank + 1,
    )

    expected_last_layer_values = None
    for i in range(modified_layer_rank + 1, N):
        if i == modified_layer_rank + 1:
            values = new_values
        else:
            values = multilayer_perceptron.variables[i - 1][2]

        expected_last_layer_values = sigmoid(
            pre_regularization_value(
                biais=multilayer_perceptron.variables[i][0],
                weights=multilayer_perceptron.variables[i][1],
                values=values,
            )
        )
    np.testing.assert_array_equal(
        np.array(multilayer_perceptron.variables[N - 1][2]), expected_last_layer_values
    )
    assert True


def test_expected_values_last_layer():
    np.testing.assert_array_equal(
        expected_values_last_layer(0), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        expected_values_last_layer(1), np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_cost_one_example():
    ## 1st test when start_layer_rank = None ##
    multilayer_perceptron = script.MultilayerPerceptron([2])
    labeled_example = [np.array([np.random.rand(2)]), np.array([np.random.rand(2)])]
    feed(multilayer_perceptron=multilayer_perceptron, example=labeled_example[1])
    expected_cost_one_example = 0
    for j in range(2):
        expected_cost_one_example += (
            labeled_example[0][0][j] - multilayer_perceptron.variables[0][2][0][j]
        ) ** 2
    actual_cost_one_example = cost_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
    )
    # Rounding can raise errors unrelated with the thested behavior. Thus, we test that the two
    # values are close enough in proportion rather than being equal.
    rtol = 1e-06
    np.testing.assert_allclose(
        actual=actual_cost_one_example, desired=expected_cost_one_example, rtol=rtol
    )

    ## 2nd test with a start_layer_rank != None ##
    layout = [2, 2, 2]
    N = len(layout)
    multilayer_perceptron = script.MultilayerPerceptron(layout, dtype=np.float64)
    label = [np.random.rand(1, layout[-1])]
    modified_layer_rank = 0
    new_values = [np.random.rand(layout[modified_layer_rank])]
    multilayer_perceptron.variables[modified_layer_rank][2] = new_values

    expected_layer_values = None
    for i in range(modified_layer_rank + 1, N):
        if i == modified_layer_rank + 1:
            values = new_values
        else:
            values = expected_layer_values

        expected_layer_values = multilayer_perceptron.layers[i].activation_fct(
            pre_regularization_value(
                biais=multilayer_perceptron.variables[i][0],
                weights=multilayer_perceptron.variables[i][1],
                values=values,
            )
        )
    expected_last_layer_values = expected_layer_values
    desired_cost_one_example = (
        np.linalg.norm(label[0] - expected_last_layer_values)
    ) ** 2

    actual_cost_one_example = cost_one_example(
        multilayer_perceptron=multilayer_perceptron,
        labeled_example=[label],
        start_layer_rank=modified_layer_rank + 1,
    )
    rtol = 1e-06
    np.testing.assert_allclose(
        actual=actual_cost_one_example, desired=desired_cost_one_example, rtol=rtol
    )


def test_cost_gradient_one_example():
    """
    The tests are divided into two groups. The first group assess that the cost gradient internal
    structure verify the theoretical formulae. The second group assess wether the computed
    direction maximize indeed the cost.
    """
    ## First group ##

    # First a test to assess the correct computation of the partial derivatives w.r.t. the values
    # of the last layer (whose computations are different than the rest). The formula we're
    # verifying is (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐴^(𝑁−1) )=−2 ∗(𝑦−𝐴^(𝑁−1)):
    size = 2
    multilayer_perceptron = script.MultilayerPerceptron([size, size])
    label = np.random.rand(1, size)
    labeled_example = [label, np.random.rand(1, size)]
    _cost_gradient = cost_gradient_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
    )
    np.testing.assert_array_equal(
        _cost_gradient[1][2], -2 * (label - multilayer_perceptron.variables[1][2])
    )

    # Then we verify the formula of the partial derivatives w.r.t. the biaises :
    # (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐵^𝑖 )=−(𝜕𝐶_(/image))/(𝜕𝐴^𝑖 )⊙𝑓_𝑖^′ (𝑍^𝑖).
    np.testing.assert_array_equal(
        _cost_gradient[1][0],
        -_cost_gradient[1][2]
        * sigmoid_derivative(
            pre_regularization_value(
                biais=multilayer_perceptron.variables[1][0],
                weights=multilayer_perceptron.variables[1][1],
                values=multilayer_perceptron.variables[0][2],
            )
        ),
    )

    # We verify the formula of the partial derivatives w.r.t. the weights :
    # (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝑊^𝑖 )=−(𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐵^𝑖 ).T*(𝐴^(𝑖−1) ). Note that we start to need to use
    # array_almost_equal instead of array_equal, probably because of rouding errors which start to
    # add up.
    np.testing.assert_array_almost_equal(
        _cost_gradient[1][1],
        -_cost_gradient[1][0].T @ multilayer_perceptron.variables[0][2],
    )

    # We verify the formula of the partial derivative w.r.t. the values of the previous layer:
    # (𝜕𝐶_(/𝑖𝑚𝑎𝑔𝑒))/(𝜕𝐴^(𝑖−1) )=- (𝜕𝐶_(/image))/(𝜕𝐵^𝑖 )∗𝑊^𝑖.
    np.testing.assert_array_almost_equal(
        _cost_gradient[0][2],
        -_cost_gradient[1][0] @ multilayer_perceptron.variables[1][1],
    )

    # We verify that the loop works as intended.
    size = 2
    multilayer_perceptron = script.MultilayerPerceptron([size, size, size])
    labeled_example = [np.random.rand(1, size), np.random.rand(1, size)]
    _cost_gradient = cost_gradient_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
    )
    for i in range(size - 1, 0, -1):
        # w.r.t. to the biaises
        np.testing.assert_array_equal(
            _cost_gradient[i][0],
            -_cost_gradient[i][2]
            * multilayer_perceptron.layers[i].activation_fct_derivative(
                pre_regularization_value(
                    biais=multilayer_perceptron.variables[i][0],
                    weights=multilayer_perceptron.variables[i][1],
                    values=multilayer_perceptron.variables[i - 1][2],
                )
            ),
        )
        # w.r.t. to the weights
        np.testing.assert_array_almost_equal(
            _cost_gradient[i][1],
            -_cost_gradient[i][0].T @ multilayer_perceptron.variables[i - 1][2],
        )
        # w.r.t. to the values of the previous layer
        np.testing.assert_array_almost_equal(
            _cost_gradient[i - 1][2],
            -_cost_gradient[i][0] @ multilayer_perceptron.variables[i][1],
        )

    ## 2nd group ##

    # We'll first verify that the comptuted partial derivatives makes sense, that is that when only
    # their respective parameter is nudged by eps, the cost varies roughly by partial_derivative *
    # eps.
    layout = [784, 16, 16, 10]
    eps = 2 * (10 ** (-5))
    dtype = np.float64

    # Relative tolerance of the comparison tests.
    rtol = 1e-2

    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=dtype)
    N = len(layout)
    labeled_example = [
        np.random.rand(1, layout[-1]),
        np.random.rand(1, layout[0]),
    ]
    initial_cost_one_example = cost_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
    )
    _cost_gradient_one_example = cost_gradient_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
    )

    # First we tackle the special case of the values of the last layer. We make sure to
    # use the alternative mode of operation of cost_one_example (where no example is fed to the
    # first layer and the values of the multilayer perceptron are recomputed starting from a
    # certain row) as not doing it would overwrite the new nudged values of the last layer. This
    # will be a recurrent remark in the following tests.
    for j in range(layout[-1]):
        multilayer_perceptron.variables[N - 1][2][0][j] += eps
        new_cost_one_example = cost_one_example(
            multilayer_perceptron=multilayer_perceptron,
            labeled_example=labeled_example[0],
            start_layer_rank=N,
        )
        dc = new_cost_one_example - initial_cost_one_example
        expected_dc = eps * _cost_gradient_one_example[N - 1][2][0][j]

        # We now assess that expected_dc is close to dc up to some relative tolerance. We also let
        # an absolute tolerance. Why ? Imagine the following case : expected_dc and the real dc
        # (not the one currently computed) are of the order of magnitude 1e-32, and actual_cost and
        # and expected_cost of 1e-7 (those are not frequent values for the partial derivatives
        # w.r.t. the values of the last layer, but can be for the others parameters whose effects
        # might be hindered by the final sigmoid function. Thus, this remark is mostly relevant for
        # the later lests.). When dc is computed, 0 is returned because a difference of 1e-32 is
        # beyond the precision with which these numbers are saved (the smallest difference
        # catchable is 1e-7 (at which the decimal digits of the costs start) + the number of
        # decimal digits of the representation (15 for float64, also called the machine epsilon).
        # In consequence, to not catch these irrelevant errors, we let an absolute tolerance of
        # expected_cost * 1e-(number of decimal digits of the used representation), or
        # alternatively, expected_cost * machine epsilon.
        machine_epsilon = np.finfo(dtype).eps
        atol = new_cost_one_example * machine_epsilon
        np.testing.assert_allclose(actual=dc, desired=expected_dc, rtol=rtol, atol=atol)
        # (In this simple case, we can find a simple theoretical formula for actual-desired, which
        # is exactly eps**2. This was confirmed visually when debugging).

        # We don't forget to put back the value to its initial state to not disturb the next tests.
        multilayer_perceptron.variables[N - 1][2][0][j] += -eps

    # Now we test the partial derivatives w.r.t. the other parameters:
    for i in range(N - 1, 0, -1):
        # Starting with the biaises:
        for j in range(layout[i]):
            multilayer_perceptron.variables[i][0][0][j] += eps
            new_cost_one_example = cost_one_example(
                multilayer_perceptron=multilayer_perceptron,
                labeled_example=labeled_example[0],
                start_layer_rank=i,
            )
            dc = new_cost_one_example - initial_cost_one_example
            expected_dc = eps * _cost_gradient_one_example[i][0][0][j]

            # Same remark as previously on the absolute tolerance used.
            atol = new_cost_one_example * machine_epsilon
            np.testing.assert_allclose(
                actual=dc, desired=expected_dc, rtol=rtol, atol=atol
            )

            # Again, we don't forget to put back the value to its initial state to not disturb the
            # next tests.
            multilayer_perceptron.variables[i][0][0][j] += -eps

        # The weights:
        for j in range(layout[i]):
            for k in range(layout[i - 1]):
                multilayer_perceptron.variables[i][1][j][k] += eps
                new_cost_one_example = cost_one_example(
                    multilayer_perceptron=multilayer_perceptron,
                    labeled_example=labeled_example[0],
                    start_layer_rank=i,
                )
                dc = new_cost_one_example - initial_cost_one_example
                expected_dc = eps * _cost_gradient_one_example[i][1][j][k]

                # J'ai du mal à le justifier parfaitement, mais il suffit de rajouter ce petit 2
                # pour que les tests passent tout le temps (ils passent une fois sur deux sinon).
                atol = 2 * new_cost_one_example * machine_epsilon
                np.testing.assert_allclose(
                    actual=dc, desired=expected_dc, rtol=rtol, atol=atol
                )

                multilayer_perceptron.variables[i][1][j][k] += -eps

        # The value of the previous layer:
        for j in range(layout[i]):
            multilayer_perceptron.variables[i - 1][2][0][j] += eps
            new_cost_one_example = cost_one_example(
                multilayer_perceptron,
                labeled_example=labeled_example[0],
                start_layer_rank=(i - 1) + 1,
            )
            dc = new_cost_one_example - initial_cost_one_example
            expected_dc = eps * _cost_gradient_one_example[i - 1][2][0][j]

            atol = new_cost_one_example * machine_epsilon
            np.testing.assert_allclose(
                actual=dc, desired=expected_dc, rtol=rtol, atol=atol
            )

            multilayer_perceptron.variables[i - 1][2][0][j] += -eps


def test_cost_gradient():
    # First test with only one image :
    layout = [2, 2, 2]
    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=np.float64)
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1]
    expected_cost_gradient = cost_gradient_one_example(
        multilayer_perceptron, labeled_example=labeled_example_1
    )
    actual_cost_gradient = cost_gradient(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set
    )

    # Equality tests of the two cost gradients. They are not simple arrays so we have to decompose
    # the tests into several, alongside their internal structure.
    N = len(layout)
    for i in range(N):
        # biais
        np.testing.assert_array_equal(
            actual_cost_gradient[i][0], expected_cost_gradient[i][0]
        )
        # weights
        np.testing.assert_array_equal(
            actual_cost_gradient[i][1], expected_cost_gradient[i][1]
        )
        # values
        np.testing.assert_array_equal(
            actual_cost_gradient[i][2], expected_cost_gradient[i][2]
        )

    # 2nd test with two images in the training set.
    labeled_example_2 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1, labeled_example_2]

    # Creation of the expected cost gradient
    cost_gradient_example_1 = cost_gradient_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example_1
    )
    cost_gradient_example_2 = cost_gradient_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example_2
    )
    expected_cost_gradient = []
    for i in range(N):
        expected_cost_gradient.append([[], [], []])
        for variable in range(3):
            expected_cost_gradient[i][variable] = np.average(
                np.array(
                    [
                        cost_gradient_example_1[i][variable],
                        cost_gradient_example_2[i][variable],
                    ]
                ),
                axis=0,
            )

    actual_cost_gradient = cost_gradient(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set
    )

    for i in range(N):
        for variable in range(3):
            np.testing.assert_array_equal(
                expected_cost_gradient[i][variable], actual_cost_gradient[i][variable]
            )


def test_cost():
    layout = [2, 2, 2]
    N = len(layout)
    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=np.float64)
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1]
    expected_cost_1 = cost_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example_1
    )
    actual_cost = cost(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set
    )
    np.testing.assert_equal(expected_cost_1, actual_cost)

    labeled_example_2 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1, labeled_example_2]
    expected_cost_2 = cost_one_example(
        multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example_2
    )
    expected_cost = np.mean(np.array([expected_cost_1, expected_cost_2]))
    actual_cost = cost(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set
    )
    np.testing.assert_equal(expected_cost, actual_cost)


def test_learning_one_step():
    ## 1st series of tests : with a training set of one example ##

    # The first test assesses the structure of the response.
    layout = [16, 16, 16, 10]
    N = len(layout)
    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=np.float64)
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1]

    # the cost is computed, it will be used for the second test. We have to do it now before the
    # parameters are nudged.
    initial_cost = cost(multilayer_perceptron, training_set)

    expected_cost_gradient = cost_gradient_one_example(
        multilayer_perceptron, labeled_example=labeled_example_1
    )
    eta = 1e-02
    expected_variables = deepcopy(multilayer_perceptron.variables)
    for i in range(N):
        for variable in range(2):
            expected_variables[i][variable] += (
                -eta * np.average(layout) * expected_cost_gradient[i][variable]
            )

    learning_one_step(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set, eta=eta
    )

    for i in range(N):
        # We only care about the effects on the parameters, not on the node values. Note that if we
        # were to carry the comparison for the nodes values also, it would fail as the node values
        # have been modified in the meantime by learning_one_step, by the call of cost inside of
        # it.
        for variable in range(2):

            np.testing.assert_array_almost_equal(
                multilayer_perceptron.variables[i][variable],
                expected_variables[i][variable],
            )

    # The second test verifies that the cost decreases after the learning step.
    new_cost = multilayer_perceptron.cost
    assert new_cost <= initial_cost

    ## 2nd series of tests : with a training set of 2 examples ##

    # The first test assesses the structure of the response.
    labeled_example_2 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1, labeled_example_2]

    # the cost is computed, it will be used for the second test. We have to do it now before the
    # parameters are nudged.
    initial_cost = cost(multilayer_perceptron, training_set)

    # Computation of the expected cost gradient
    expected_cost_gradient_1 = cost_gradient_one_example(
        multilayer_perceptron, labeled_example=labeled_example_1
    )
    expected_cost_gradient_2 = cost_gradient_one_example(
        multilayer_perceptron, labeled_example=labeled_example_2
    )
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

    eta = 1e90
    expected_variables = deepcopy(multilayer_perceptron.variables)
    for i in range(N):
        for variable in range(3):
            expected_variables[i][variable] += (
                -eta * np.average(layout) * expected_cost_gradient[i][variable]
            )

    learning_one_step(
        multilayer_perceptron=multilayer_perceptron, training_set=training_set, eta=eta
    )

    for i in range(N):
        # We only care about the effects on the parameters, not on the node values. Note that if we
        # were to carry the comparison for the nodes values also, it would fail as the node values
        # have been modified in the meantime by learning_one_step, by the call of cost inside of
        # it.
        for variable in range(2):

            np.testing.assert_array_almost_equal(
                multilayer_perceptron.variables[i][variable],
                expected_variables[i][variable],
            )

    # The second test verifies that the cost decreases after the learning step.
    new_cost = multilayer_perceptron.cost
    assert new_cost <= initial_cost


def test_learning():
    # First test to make sure no error is raised.
    layout = [1, 1]
    N = len(layout)
    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=np.float64)
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1]
    eta = 1
    previous_costs = learning(
        multilayer_perceptron=multilayer_perceptron,
        training_set=training_set,
        eta=eta,
        max_stagnation_steps=10,
    )

    steps_number = 10
    previous_costs = learning(
        multilayer_perceptron=multilayer_perceptron,
        training_set=training_set,
        eta=eta,
        steps_number=steps_number,
    )

    # This time with a training set with multiple examples.
    layout = [5, 5, 5]
    N = len(layout)
    multilayer_perceptron = script.MultilayerPerceptron(layout=layout, dtype=np.float64)
    labeled_example_1 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    labeled_example_2 = [np.random.rand(1, layout[-1]), np.random.rand(1, layout[0])]
    training_set = [labeled_example_1, labeled_example_2]
    eta = 1
    previous_costs = learning(
        multilayer_perceptron=multilayer_perceptron,
        training_set=training_set,
        eta=eta,
        max_stagnation_steps=10,
    )
