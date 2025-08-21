# ----------------------------------------
# Main Training Script
# ----------------------------------------
if __name__ == "__main__":

    with open("./config/physichal_paramters.yaml", "r") as file:
        params = yaml.safe_load(file)
    
    with open("./config/initial_conditions.yaml", "r") as file:
        init_vals = yaml.safe_load(file)

    # Assign to variables from config
    pH = init_vals["pH"]
    c_pfas_init = init_vals["c_pfas_init"] # PFAS initial concentration (mol/L)
    c_cl = init_vals["c_cl"] # chloride concentration (mol/L)
    c_so3 = init_vals["c_so3"] # sulfite concentration (mol/L)

    
    # --- Set initital guess of constants ---
    k1, k2, k3, k4, k5, k6, k7 = [np.float32(v) for v in
                                  [5.259223e+06, 5.175076e+08, 5.232472e+08, 5.551598e+08, 5.748630e+08, 5.647780e+08,
                                   5.005279e+08]]

    # --- Data Loading and Preparation ---
    df = pd.read_csv('./data/Batch_PFAS_data.csv')  # CSV columns: sequence_id, time (s), C7F15COO-, C5F11COO-, C3F7COO-, C2F5COO-, CF3COO-
    groups = df.groupby("sequence_id")
    columns = ["C7F15COO-", "C5F11COO-", "C3F7COO-", "C2F5COO-", "CF3COO-"]

    t_true_list = []       # experimental time grids (unpadded), one per sequence
    y_true_list = []       # experimental outputs (each: (T_exp, 5))
    initial_states_list = []  # each initial state: (8,)

    for seq_id, group in groups:
        group_sorted = group.sort_values(by="time (s)")
        t_seq = group_sorted["time (s)"].values   # shape (T_exp,)
        y_seq = group_sorted[columns].values       # shape (T_exp, 5)
        t_true_list.append(t_seq)
        y_true_list.append(y_seq)
        init_state = np.zeros((8,), dtype=np.float32)
        init_state[0] = y_seq[0, 0]
        initial_states_list.append(init_state)

    batch_size = len(t_true_list)

    # --- Build Fine-Grained Simulation Grids ---
    t_pinn_list = []     # For each sequence, create a fine-grained simulation grid from its start to its end time using dt_sim.
    T_sim_list = []

    for t_seq in t_true_list:
        # Create a fine grid from the first to the last experimental time
        t_sim = np.arange(t_seq[0], t_seq[-1] + dt_sim, dt_sim)
        t_pinn_list.append(t_sim)
        T_sim_list.append(len(t_sim))
    T_sim_max = max(T_sim_list)

    # --- Prepare Dummy Input for Simulation ---
    dummy_input = np.zeros((batch_size, T_sim_max, 1), dtype=np.float32)     # Dummy input must have shape (batch, T_sim_max, 1)
    dummy_input = tf.convert_to_tensor(dummy_input, dtype=tf.float32)

    # --- Pad Experimental Data for Loss ---
    T_exp_list = [t.shape[0] for t in t_true_list]     # We need to pad the experimental outputs to a common length T_exp_max.
    T_exp_max = max(T_exp_list)
    y_true_padded = []
    t_true_padded = []
    for t_seq, y_seq in zip(t_true_list, y_true_list):
        pad_len = T_exp_max - t_seq.shape[0]
        t_seq_pad = np.pad(t_seq, (0, pad_len), mode='constant', constant_values=0)
        y_seq_pad = np.pad(y_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        t_true_padded.append(t_seq_pad)
        y_true_padded.append(y_seq_pad)
    y_train = tf.convert_to_tensor(np.stack(y_true_padded, axis=0), dtype=tf.float32)

    # --- Prepare Initial States ---
    initial_states = tf.convert_to_tensor(np.stack(initial_states_list, axis=0), dtype=tf.float32)

    # --- Create tf.data.Dataset ---
    dataset = tf.data.Dataset.from_tensor_slices(((dummy_input, initial_states), y_train))     # Each element is ((dummy_input_i, initial_state_i), y_train_i)
    dataset = dataset.batch(batch_size)     # Set batch size to full dataset or a chosen batch size

    # Use system script for traning
    for_prediction = False

    # --- Create Model ---
    model = create_model(k1, k2, k3, k4, k5, k6, k7,  params, c_cl, c_so3, pH, dt_sim, initial_states, t_pinn_list, t_true_list,for_prediction)

    # --- Training ---
    print("Before training, predictions:")
    y_pred_before = model.predict([dummy_input, initial_states])
    start_time = time.time()
    model.fit(dataset, epochs=200, verbose=1)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    # Save trained weights
    model.save_weights("./checkpoints/pinn_model.weights.h5")
    y_pred = model.predict([dummy_input, initial_states])

    t_sim_seq0 = t_pinn_list[1]  # This is your fine simulation grid for sequence 0

    # Extract raw fine-grained prediction for the first sequence.
    y_pred_first = y_pred[0:1, :, :]  # shape: (1, T_sim_max, 5)
    y_pred_first = y_pred_first.squeeze(0)  # shape: (T_sim_max, 5)
    y_pred_second = y_pred[1:2, :, :]
    y_pred_second = y_pred_second.squeeze(0)


    plt.figure(figsize=(9, 6))
    outputs_to_plot = [0, 1, 2, 3, 4]
    labels = ['C7F15COO-', 'C5F11COO-', 'C3F7COO-', 'C2F5COO-', 'CF3COO-']

    for i, label in enumerate(labels):
        plt.subplot(3, 2, i + 1)
        plt.scatter(t_true_list[0], y_true_list[0][:, i], color='gray', label='Raw Data batch 1', marker='o', alpha=0.7)
        plt.scatter(t_true_list[1], y_true_list[1][:, i], color='gray', label='Raw Data batch 2', marker='o', alpha=0.7)
        plt.plot(t_sim_seq0, y_pred_first[:, i], 'b', label='After Training batch 1')
        plt.plot(t_sim_seq0, y_pred_second[:,i], 'r', label='After Training batch 2')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/HPINN_minimal_model_exp_batch2.png", dpi=600, bbox_inches='tight')
    plt.show()

    # --- Print Trained Parameters ---
    rk_cell_trained = model.rnn.cell  # Access the custom cell inside the RNN layer.
    trained_params = {name: 10 ** rk_cell_trained.log_k_values[name].numpy() 
                      for name in rk_cell_trained.log_k_values}
    print("Trained parameters:")
    for name, value in trained_params.items():
        print(f"{name}: {value:.6e}")
    
    trained_params = {
        name: float(10 ** rk_cell_trained.log_k_values[name].numpy())
        for name in rk_cell_trained.log_k_values
    }

    with open("./config/trained_params.yaml", "w") as file:
        yaml.dump(trained_params, file, default_flow_style=False)