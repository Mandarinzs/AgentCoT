from agentcot.data.splitter import split_dataset


def test_split_dataset_sizes() -> None:
    data = [{"id": i} for i in range(10)]
    train, val, test = split_dataset(data, train_ratio=0.6, val_ratio=0.2, seed=1)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
