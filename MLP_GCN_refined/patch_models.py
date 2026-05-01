import os
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Part 1
    old1 = """with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2", "Fold", "Total_Params", "Trainable_Params"])

for fold, (train_idx, val_idx) in enumerate("""
    
    new1 = """with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_MSE", "Val_MSE", "Val_RMSE", "Val_MAE", "Val_R2", "Fold", "Total_Params", "Trainable_Params"])

best_overall_val_mse = float('inf')
best_model_state = None

for fold, (train_idx, val_idx) in enumerate("""

    if old1 in content and "best_overall_val_mse = float('inf')" not in content:
        content = content.replace(old1, new1)

    # Part 2 (GCN files and MLP file have slightly different formatting for "if v_mse < best_val_mse - 0.001:" )
    old2_mlp = """        if v_mse < best_val_mse - 0.001:
            best_val_mse = v_mse
            # Snapshot metrics
            best_mae = v_mae
            best_r2 = v_r2
            best_tr_mse = tr_loss.item()
            
        if (epoch + 1) % 50 == 0:"""
        
    new2_mlp = """        if v_mse < best_val_mse - 0.001:
            best_val_mse = v_mse
            # Snapshot metrics
            best_mae = v_mae
            best_r2 = v_r2
            best_tr_mse = tr_loss.item()
            if v_mse < best_overall_val_mse:
                best_overall_val_mse = v_mse
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 50 == 0:"""

    if old2_mlp in content:
        content = content.replace(old2_mlp, new2_mlp)

    old2_gcn_1 = """        if val_loss < best_val_mse - 0.001:
            best_val_mse = val_loss
            patience_cnt = 0
            best_mae = v_mae
            best_r2 = v_r2
        else:"""

    new2_gcn_1 = """        if val_loss < best_val_mse - 0.001:
            best_val_mse = val_loss
            patience_cnt = 0
            best_mae = v_mae
            best_r2 = v_r2
            if val_loss < best_overall_val_mse:
                best_overall_val_mse = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:"""

    if old2_gcn_1 in content:
        content = content.replace(old2_gcn_1, new2_gcn_1)

    old2_gcn_2 = """        if val_loss < best_val_mse - 0.001:
            best_val_mse = val_loss
            best_mae = v_mae
            best_r2 = v_r2
            
        if (epoch + 1) % 50 == 0:"""

    new2_gcn_2 = """        if val_loss < best_val_mse - 0.001:
            best_val_mse = val_loss
            best_mae = v_mae
            best_r2 = v_r2
            if val_loss < best_overall_val_mse:
                best_overall_val_mse = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 50 == 0:"""

    if old2_gcn_2 in content:
        content = content.replace(old2_gcn_2, new2_gcn_2)

    # Part 3
    old3 = """print(f"\\nTraining logs saved to: {csv_filename}")

print("\\n" + "="*40)"""

    new3 = """print(f"\\nTraining logs saved to: {csv_filename}")

if best_model_state is not None:
    model_filename = csv_filename.replace('.csv', '.pth')
    torch.save(best_model_state, model_filename)
    print(f"Best model weights saved to: {model_filename}")

print("\\n" + "="*40)"""

    if old3 in content:
        content = content.replace(old3, new3)

    old4 = """print(f"\\nTraining logs saved to: {csv_filename}")

# ═══════════════════════════════════════════════════════════════════════"""

    new4 = """print(f"\\nTraining logs saved to: {csv_filename}")

if best_model_state is not None:
    model_filename = csv_filename.replace('.csv', '.pth')
    torch.save(best_model_state, model_filename)
    print(f"Best model weights saved to: {model_filename}")

# ═══════════════════════════════════════════════════════════════════════"""

    if old4 in content:
        content = content.replace(old4, new4)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Patched {filepath}")

for f in glob.glob("MLP/GCN-refined/*.py"):
    if f.endswith("patch_models.py"): continue
    patch_file(f)
