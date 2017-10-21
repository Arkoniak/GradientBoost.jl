module TestML

using ..CustomTS
using Base.Test

importall GradientBoost.ML
importall GradientBoost.LossFunctions

@testset CustomTestSet "Machine Learning API" begin
  @testset CustomTestSet "not implemented functions throw an error" begin
    gbl = GBLearner(GBDT{Float64}(), :regression)
    instances = 1
    labels = 1

    @test_throws ErrorException fit!(gbl, instances, labels)
    @test_throws ErrorException predict!(gbl, instances)
  end

    instances = [
      1.0 1.0;
      1.0 8.0;
      1.0 10.0
    ]
    labels = [
      0.0;
      1.0;
      1.0;
    ]

  @testset CustomTestSet "fit! on Float64 arrays works" begin
    gbl = GBLearner(GBDT{Float64}(), :regression)
    @test gbl.model == nothing
    fit!(gbl, instances, labels)
    @test gbl.model != nothing
  end

  @testset CustomTestSet "predict! on Float64 arrays works" begin
    gbl = GBLearner(GBDT{Float64}(;loss_function=BinomialDeviance()), :class)
    fit!(gbl, instances, labels)
    predictions = predict!(gbl, instances)
    @test eltype(predictions) == Float64
  end

  @testset CustomTestSet "logistic works" begin
    x = [-Inf, -1, 0, 1, Inf]
    expected = [0.0, 0.0, 0.0, 1.0, 1.0]

    actual = round.(ML.logistic(x))
    @test actual == expected
  end

  @testset CustomTestSet "postprocess_pred works" begin
    predictions = [-Inf, 0.0, Inf]
    expected = [0.0, 0.0, 1.0]
    actual = ML.postprocess_pred(:class, BinomialDeviance(), predictions)
    @test actual == expected

    predictions = [-Inf, 0.0, Inf]
    actual = ML.postprocess_pred(:class_prob, BinomialDeviance(), predictions)
    @test all(i -> (0 <= i <= 1), actual)

    predictions = [-Inf, 0.0, Inf]
    expected = predictions
    actual = ML.postprocess_pred(:regression, LeastSquares(), predictions)
    @test actual == expected

    predictions = [-Inf, 0.0, Inf]
    @test_throws ErrorException ML.postprocess_pred(:class, LeastSquares(), predictions)
  end
end

end # module
