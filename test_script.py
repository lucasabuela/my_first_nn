import numpy as np
import script
from script import (
    relu,
    sigmoid,
    sigmoid_derivative,
    pre_regularization_value,
    feed,
    expected_values_last_layer,
    cost_gradient_one_example,
    cost_one_example,
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


def test_expected_values_last_layer():
    np.testing.assert_array_equal(
        expected_values_last_layer(0), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        expected_values_last_layer(1), np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_cost_one_example():
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
    layout = [1, 1, 1]
    eps = 2 * (10 ** (-4))
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

    # First we tackle the special case of the values of the last layer. We can't use the
    # cost_one_example to compute the cost in this case as it would change the values of
    # the last layer.
    for j in range(layout[-1]):
        multilayer_perceptron.variables[N - 1][2][0][j] += eps
        new_cost_one_example = (
            np.linalg.norm(
                labeled_example[0][0] - multilayer_perceptron.variables[N - 1][2][0]
            )
            ** 2
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
                labeled_example=labeled_example,
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
                    labeled_example=labeled_example,
                )
                dc = new_cost_one_example - initial_cost_one_example
                expected_dc = eps * _cost_gradient_one_example[i][1][j][k]

                atol = new_cost_one_example * machine_epsilon
                np.testing.assert_allclose(
                    actual=dc, desired=expected_dc, rtol=rtol, atol=atol
                )

                multilayer_perceptron.variables[i][1][j][k] += -eps

        # The value of the previous layer:
        for j in range(layout[i]):
            multilayer_perceptron.variables[i - 1][2][0][j] += eps
            new_cost_one_example = cost_one_example(
                multilayer_perceptron, labeled_example=labeled_example
            )
            dc = new_cost_one_example - initial_cost_one_example
            expected_dc = eps * _cost_gradient_one_example[i - 1][2][0][j]

            atol = new_cost_one_example * machine_epsilon
            np.testing.assert_allclose(
                actual=dc, desired=expected_dc, rtol=rtol, atol=atol
            )

            multilayer_perceptron.variables[i - 1][2][0][j] += -eps

    assert True

    """
    nb_tests = 10
    # We'll verify that in nb_tests over a small nn, the cost decreases in the direction opposite
    # of the gradient. Since is only true in the linearized case. Thus, to smooth the cost function
    # (w.r.t. the parameters), we use a nn with a lot of parameters than for regular unit tests.
    nn_size = 10
    multilayer_perceptron = script.MultilayerPerceptron(
        layout=[nn_size] * 4, dtype=np.float64
    )
    for i in range(nb_tests):
        print(i)
        labeled_example = [np.random.rand(1, nn_size), np.random.rand(1, nn_size)]
        initial_cost_one_example = cost_one_example(
            multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
        )
        _cost_gradient_one_example = cost_gradient_one_example(
            multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
        )
        print(
            f"initial multilayer_perceptron.variables[1][0] = {multilayer_perceptron.variables[1][0]}"
        )
        print(f"_cost_gradient_one_example[1][0] = {_cost_gradient_one_example[1][0]}")
        # We nudge the nn parameters into the direction opposite the gradient:
        nudge_strenght = 10 ** (-1) * nn_size
        for i, layer_variables in enumerate(multilayer_perceptron.variables):
            layer_variables[0] += -nudge_strenght * _cost_gradient_one_example[i][0]
            layer_variables[1] += -nudge_strenght * _cost_gradient_one_example[i][1]

        print(
            f"new multilayer_perceptron.variables[1][0] = {multilayer_perceptron.variables[1][0]}"
        )
        new_cost_one_example = cost_one_example(
            multilayer_perceptron=multilayer_perceptron, labeled_example=labeled_example
        )
        print(
            f"initial cost = {initial_cost_one_example} & new cost = {new_cost_one_example}"
        )
        np.testing.assert_array_less(new_cost_one_example, initial_cost_one_example)"""
