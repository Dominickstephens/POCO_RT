# Suggested rewrite for poco_tensorrt_converter.py
import torch
import torch_tensorrt
from pocolib.core.tester import POCOTester
import argparse
import os
import logging

# --- Wrapper Class (Modified Forward Signature) ---
class POCOTracerWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        # Store the keys you want to return from the original model's output dict
        self.output_keys = [
            'pred_pose',      # Example: (batch, 24, 3, 3) rotation matrices
            'pred_shape',     # Example: (batch, 10) shape parameters
            'pred_cam',       # Example: (batch, 3) camera parameters
            'smpl_vertices',  # Example: (batch, 6890, 3) mesh vertices
        ]
        logging.info(f"Tracer wrapper will return tensors for keys: {self.output_keys}")
        # Store the keys expected as input in the correct order for the new forward signature
        self.input_keys = [
            'img', 'bbox_info', 'focal_length', 'scale', 'center', 'orig_shape'
        ]

    # Modified forward signature to accept individual tensors
    def forward(self, img, bbox_info, focal_length, scale, center, orig_shape):
        # Reconstruct the batch dictionary inside the wrapper
        batch = {
            'img': img,
            'bbox_info': bbox_info,
            'focal_length': focal_length,
            'scale': scale,
            'center': center,
            'orig_shape': orig_shape,
        }

        # Call the original model
        original_output_dict = self.original_model(batch)

        # Extract the desired tensors based on the predefined keys
        output_list = []
        for key in self.output_keys:
            if key in original_output_dict and isinstance(original_output_dict[key], torch.Tensor):
                output_list.append(original_output_dict[key])
            else:
                logging.warning(f"Key '{key}' not found or not a Tensor in model output during trace.")
                raise TypeError(f"Expected key '{key}' to be a Tensor in model output, but found {type(original_output_dict.get(key))}")

        # Return the tensors as a tuple
        return tuple(output_list)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert POCO PyTorch models to TensorRT-optimized TorchScript.")

    # --- Model/Config Arguments ---
    parser.add_argument('--cfg', type=str, default='configs/demo_poco_cliff.yaml',
                        help='Path to the configuration file (.yaml) defining model hyperparameters.')
    parser.add_argument('--ckpt', type=str, default='data/poco_cliff.pt',
                        help='Path to the PyTorch model checkpoint (.pt).')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save the optimized TensorRT model.')
    parser.add_argument('--output_name', type=str, default='poco_model_trt.pt',
                        help='Filename for the optimized TensorRT model.')

    # --- Arguments potentially needed by POCOTester initialization ---
    parser.add_argument('--display', action='store_true', help='display intermediate results (OpenCV window)')
    parser.add_argument('--smooth', action='store_true', help='smooth results (if implemented in POCO Tester)')
    parser.add_argument('--min_cutoff', type=float, default=0.004, help='one euro min cutoff')
    parser.add_argument('--beta', type=float, default=1.5, help='one euro beta')
    parser.add_argument('--no_kinematic_uncert', action='store_false',
                        help='Do not use SMPL Kinematic for uncert')
    parser.add_argument('--wireframe', action='store_true', help='render wireframes in demo display')
    parser.add_argument('--exp', type=str, default='', help='experiment description')
    parser.add_argument('--inf_model', type=str, default='best', help='select model from checkpoint')
    parser.add_argument('--skip_frame', type=int, default=1, help='skip frames')
    parser.add_argument('--dir_chunk_size', type=int, default=1000, help='dir chunk size')
    parser.add_argument('--dir_chunk', type=int, default=0, help='dir chunk index')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'], help='tracking method')
    parser.add_argument('--staf_dir', type=str, default='/path/to/pose-track-framework', help='STAF dir')
    parser.add_argument('--no_render', action='store_true', help='disable rendering video output (for video/folder modes)')
    parser.add_argument('--render_crop', action='store_true', help='Render cropped image (in demo display)')
    parser.add_argument('--no_uncert_color', action='store_true', help='No uncertainty color (in demo display)')
    parser.add_argument('--sideview', action='store_true', help='render side viewpoint (in demo display)')
    parser.add_argument('--draw_keypoints', action='store_true', help='draw 2d keypoints (in demo display)')
    parser.add_argument('--save_obj', action='store_true', help='save obj files (for video/folder modes)')

    # --- Optimization Parameters ---
    parser.add_argument('--input_height', type=int, default=224,
                        help='Input image height expected by the model.')
    parser.add_argument('--input_width', type=int, default=224,
                        help='Input image width expected by the model.')
    parser.add_argument('--opt_batch_size', type=int, default=1,
                        help='Optimal batch size for TensorRT optimization.')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum dynamic batch size for TensorRT.')
    parser.add_argument('--max_batch_size', type=int, default=1,
                        help='Maximum dynamic batch size for TensorRT.')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'],
                        help='TensorRT optimization precision (fp16 may require compatible hardware).')

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Sanity Checks ---
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. TensorRT requires a CUDA-enabled GPU.")
        return
    if not os.path.exists(args.cfg):
        logging.error(f"Config file not found: {args.cfg}")
        return
    if not os.path.exists(args.ckpt):
        logging.error(f"Checkpoint file not found: {args.ckpt}")
        return

    device = torch.device("cuda")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    try:
        logging.info(f"Loading model using config: {args.cfg} and checkpoint: {args.ckpt}")
        tester = POCOTester(args)
        model = tester.model
        model.eval()
        model.to(device)
        logging.info("Original model loaded successfully.")

        # Instantiate the Wrapper
        wrapper_model = POCOTracerWrapper(model)
        wrapper_model.eval()
        logging.info("POCO Tracer Wrapper created.")

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # --- Prepare Dummy Input Tensors ---
    bs = args.opt_batch_size
    img_h = args.input_height
    img_w = args.input_width

    dummy_img = torch.randn(bs, 3, img_h, img_w, dtype=torch.float32).to(device)
    dummy_bbox_info = torch.randn(bs, 3, dtype=torch.float32).to(device)
    dummy_focal_length = torch.randn(bs, dtype=torch.float32).to(device)
    dummy_scale = torch.randn(bs, dtype=torch.float32).to(device)
    dummy_center = torch.randn(bs, 2, dtype=torch.float32).to(device)
    dummy_orig_shape = torch.randint(low=img_h, high=img_h*2, size=(bs, 2), dtype=torch.float32).to(device)

    # Create tuple of dummy inputs in the order expected by the wrapper's forward method
    dummy_input_tuple = (
        dummy_img,
        dummy_bbox_info,
        dummy_focal_length,
        dummy_scale,
        dummy_center,
        dummy_orig_shape,
    )
    logging.info(f"Using tuple of dummy inputs for tracing.")

    # --- Trace the WRAPPER Model ---
    logging.info("Attempting to trace the WRAPPER model...")
    try:
        # Pass the tuple of tensors here
        traced_model = torch.jit.trace(wrapper_model, dummy_input_tuple, strict=False)
        logging.info("Wrapper model traced successfully.")
    except Exception as e:
        logging.error(f"Failed to trace model: {e}")
        logging.error("Tracing can fail with dynamic control flow or if dummy input structure/types are incorrect.")
        return

    # --- Configure TensorRT Inputs ---
    # Provide specifications for *all* inputs defined in the wrapper's forward signature.
    # Assume only 'img' needs dynamic batch size, others use static shapes from dummy data.
    trt_inputs = [
        # Input spec for 'img' (dynamic batch)
        torch_tensorrt.Input(
            min_shape=[args.min_batch_size, 3, args.input_height, args.input_width],
            opt_shape=[args.opt_batch_size, 3, args.input_height, args.input_width],
            max_shape=[args.max_batch_size, 3, args.input_height, args.input_width],
            dtype=torch.float32
        ),
        # Input spec for 'bbox_info' (static shape based on dummy data)
        torch_tensorrt.Input(shape=dummy_bbox_info.shape, dtype=torch.float32),
        # Input spec for 'focal_length' (static shape)
        torch_tensorrt.Input(shape=dummy_focal_length.shape, dtype=torch.float32),
        # Input spec for 'scale' (static shape)
        torch_tensorrt.Input(shape=dummy_scale.shape, dtype=torch.float32),
        # Input spec for 'center' (static shape)
        torch_tensorrt.Input(shape=dummy_center.shape, dtype=torch.float32),
        # Input spec for 'orig_shape' (static shape)
        torch_tensorrt.Input(shape=dummy_orig_shape.shape, dtype=torch.float32),
    ]
    logging.info(f"Configuring {len(trt_inputs)} TensorRT inputs (dynamic batch for img).")


    # --- Determine Precision ---
    enabled_precisions = {torch.float32}
    if args.precision == 'fp16':
        enabled_precisions = {torch.float16}
        logging.info("Using FP16 precision.")
    else:
        logging.info("Using FP32 precision.")

    # --- Compile with TensorRT ---
    logging.info("Compiling the traced model with TensorRT...")
    try:
        trt_model = torch_tensorrt.compile(
            traced_model,
            ir="torchscript", # Explicitly set IR to torchscript
            inputs=trt_inputs, # Use the list of input specs
            enabled_precisions=enabled_precisions,
            truncate_long_and_double=True,
            # workspace_size = 1 << 30 # Example: 1GB
        )
        logging.info("TensorRT compilation successful.")
    except Exception as e:
        logging.error(f"TensorRT compilation failed: {e}")
        # Print detailed graph info if compilation fails
        try:
            logging.error("Traced model graph:\n" + str(traced_model.graph))
        except Exception as ge:
            logging.error(f"Could not print traced model graph: {ge}")
        return

    # --- Save the Optimized Model ---
    output_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        torch.jit.save(trt_model, output_path)
        logging.info(f"TensorRT-optimized model saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save the optimized model: {e}")

if __name__ == "__main__":
    main()