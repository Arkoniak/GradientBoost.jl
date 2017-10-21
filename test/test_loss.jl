module TestLossFunctions

using ..CustomTS
using Base.Test

importall GradientBoost.LossFunctions

type DummyLossFunction <: LossFunction; end

y_examples = [
  [1,1],
  [1,0],
  [0,0],
  [-1,1],
  [-1,-1]
]
y_pred_examples = [
  [1,1],
  [1,1],
  [1,1],
  [1,1],
  [1,1]
]
bern_y_examples = [
  [1,1],
  [1,0],
  [0,1],
  [0,0],
  [0,0]
]

function test_loss(lf::LossFunction, 
  y_examples, y_pred_examples, expected)

  for i = 1:size(y_examples, 1)
    @test isapprox(loss(lf, y_examples[i], y_pred_examples[i]), expected[i], atol = 1e-5)
  end
end
function test_negative_gradient(lf::LossFunction, 
  y_examples, y_pred_examples, expected)

  for i = 1:size(y_examples, 1)
    @test isapprox(negative_gradient(lf, y_examples[i], y_pred_examples[i]), expected[i], atol = 1e-5)
  end
end
function test_minimizing_scalar(lf::LossFunction, 
  y_examples, expected)

  for i = 1:size(y_examples, 1)
    @test isapprox(minimizing_scalar(lf, y_examples[i]), expected[i], atol = 1e-5)
  end
end

@testset CustomTestSet "Loss functions" begin
  @testset CustomTestSet "not implemented functions throw an error" begin
    emp_vec = Array[]
    dlf = DummyLossFunction()

    @test_throws ErrorException loss(dlf, emp_vec, emp_vec)
    @test_throws ErrorException negative_gradient(dlf, emp_vec, emp_vec)
    @test_throws ErrorException minimizing_scalar(dlf, emp_vec)
  end

  @testset CustomTestSet "LeastSquares loss works" begin
    lf = LeastSquares()
    expected = [ 0.0, 0.5, 1.0, 2.0, 4.0 ]
    test_loss(lf, y_examples, y_pred_examples, expected)
  end
  @testset CustomTestSet "LeastSquares negative_gradient works" begin
    lf = LeastSquares()
    expected = [ [0,0], [0,-1], [-1,-1], [-2,0], [-2,-2] ]
    test_negative_gradient(lf, y_examples, y_pred_examples, expected)
  end
  @testset CustomTestSet "LeastSquares minimizing_scalar works" begin
    lf = LeastSquares()
    expected = [ 1.0, 0.5, 0.0, 0.0, -1.0 ]
    test_minimizing_scalar(lf, y_examples, expected)
  end

  @testset CustomTestSet "LeastAbsoluteDeviation loss works" begin
    lf = LeastAbsoluteDeviation()
    expected = [ 0.0, 0.5, 1.0, 1.0, 2.0 ]
    test_loss(lf, y_examples, y_pred_examples, expected)
  end
  @testset CustomTestSet "LeastAbsoluteDeviation negative_gradient works" begin
    lf = LeastAbsoluteDeviation()
    expected = [ [0,0], [0,-1], [-1,-1], [-1,0], [-1,-1] ]
    test_negative_gradient(lf, y_examples, y_pred_examples, expected)
  end
  @testset CustomTestSet "LeastAbsoluteDeviation minimizing_scalar works" begin
    lf = LeastAbsoluteDeviation()
    expected = [ 1.0, 0.5, 0.0, 0.0, -1.0 ]
    test_minimizing_scalar(lf, y_examples, expected)
  end

  @testset CustomTestSet "BinomialDeviance loss works" begin
    lf = BinomialDeviance()
    expected = [ 
      0.626523, 1.626523, 1.626523, 2.626523, 2.626523,
    ]
    test_loss(lf, bern_y_examples, y_pred_examples, expected)
  end
  @testset CustomTestSet "BinomialDeviance negative_gradient works" begin
    lf = BinomialDeviance()
    expected = [ 
      [0.268941, 0.268941], 
      [0.268941, -0.731059], 
      [-0.731059, 0.268941], 
      [-0.731059, -0.731059], 
      [-0.731059, -0.731059] 
    ]
    test_negative_gradient(lf, bern_y_examples, y_pred_examples, expected)
  end
  @testset CustomTestSet "BinomialDeviance minimizing_scalar works" begin
    lf = BinomialDeviance()
    expected = [ Inf, 0.0, 0.0, -Inf, -Inf ]
    test_minimizing_scalar(lf, bern_y_examples, expected)
  end
end

end # module
