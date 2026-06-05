import pytest
import numpy as np
import pandas as pd
import torch

import pytest
import numpy as np
import pandas as pd
import torch

# Assuming your function is in data/pipeline.py
# from context_based_anomalous_flow_detector.data.pipeline import create_sequences
from context_based_anomalous_flow_detector.mini_bert.mini_bert import create_sequences
def test_create_sequences_truncation():
    """
    Verifies that create_sequences cleanly drops remainder rows
    when the total dataset length is not perfectly divisible by seq_len.
    """
    # 1. Arrange: Create a mock DataFrame with an awkward length (e.g., 50)
    # 50 rows / 34 window = 1 sequence, 16 rows should be dropped.
    SEQ_LEN = 34
    INPUT_ROWS = 50
    NUM_CONT_FEATURES = 13

    # Generate random continuous features
    mock_data = {f"col_{i}": np.random.randn(INPUT_ROWS) for i in range(NUM_CONT_FEATURES)}
    # Add categorical columns
    mock_data["portdst"] = np.random.randint(0, 65535, size=INPUT_ROWS)
    mock_data["proto"] = np.random.randint(0, 255, size=INPUT_ROWS)

    df_awkward = pd.DataFrame(mock_data)

    # 2. Act: Run your actual sequence function
    # train_cont, train_port, train_proto = create_sequences(df_awkward, seq_len=SEQ_LEN)

    # --- EXPECTED RESULTS FOR ASSERTIONS ---
    expected_sequences = INPUT_ROWS // SEQ_LEN  # 50 // 34 = 1

    # 3. Assert: Verify the shapes are strictly bound to the truncated length
    # Uncomment these once you hook up your actual function
    # assert train_cont.shape == (expected_sequences, SEQ_LEN, NUM_CONT_FEATURES)
    # assert train_port.shape == (expected_sequences, SEQ_LEN)
    # assert train_proto.shape == (expected_sequences, SEQ_LEN)

    print(f"\nSuccessfully verified: {INPUT_ROWS} rows truncated to {expected_sequences} sequence(s).")

def test_create_sequences_truncation():
    """
    Verifies that create_sequences cleanly drops remainder rows
    when the total dataset length is not perfectly divisible by seq_len.
    """
    # 1. Arrange: Create a mock DataFrame with an awkward length (e.g., 50)
    # 50 rows / 34 window = 1 sequence, 16 rows should be dropped.
    SEQ_LEN = 34
    INPUT_ROWS = 50
    NUM_CONT_FEATURES = 13

    # Generate random continuous features
    mock_data = {f"col_{i}": np.random.randn(INPUT_ROWS) for i in range(NUM_CONT_FEATURES)}
    # Add categorical columns
    mock_data["portdst"] = np.random.randint(0, 65535, size=INPUT_ROWS)
    mock_data["proto"] = np.random.randint(0, 255, size=INPUT_ROWS)

    df_awkward = pd.DataFrame(mock_data)

    # 2. Act: Run your actual sequence function
    train_cont, train_port, train_proto = create_sequences(df_awkward, seq_len=SEQ_LEN)
    print(f"{train_cont=}")
    print(f"{train_port=}")
    print(f"{train_proto=}")

    # --- EXPECTED RESULTS FOR ASSERTIONS ---
    expected_sequences = INPUT_ROWS // SEQ_LEN  # 50 // 34 = 1

    # 3. Assert: Verify the shapes are strictly bound to the truncated length
    # Uncomment these once you hook up your actual function
    assert train_cont.shape == (expected_sequences, SEQ_LEN, NUM_CONT_FEATURES)
    assert train_port.shape == (expected_sequences, SEQ_LEN)
    assert train_proto.shape == (expected_sequences, SEQ_LEN)

    print(f"\nSuccessfully verified: {INPUT_ROWS} rows truncated to {expected_sequences} sequence(s).")