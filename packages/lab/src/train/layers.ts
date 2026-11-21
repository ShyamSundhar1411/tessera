import * as tf from "@tensorflow/tfjs";

// The few layers tf.js does not ship that the engine's ops need.

/** x * relu6(x + 3) / 6, MobileNetV3's cheap stand-in for SiLU. */
export class HardSwish extends tf.layers.Layer {
  static className = "TesseraHardSwish";
  override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return tf.mul(x, tf.div(tf.relu6(tf.add(x, 3)), 6));
    });
  }
}

/**
 * Pads height and width with a constant. Max pooling with padding needs the
 * border filled with a very negative number, as PyTorch does, so the border
 * never wins; zero padding would change the answer after negative values.
 */
export class ConstantPad2d extends tf.layers.Layer {
  static className = "TesseraConstantPad2d";
  private pad: number;
  private value: number;
  constructor(config: { pad: number; value: number; name?: string }) {
    super({ name: config.name });
    this.pad = config.pad;
    this.value = config.value;
  }
  override computeOutputShape(shape: tf.Shape | tf.Shape[]): tf.Shape {
    const s = shape as tf.Shape;
    const grow = (n: number | null) => (n === null ? null : n + 2 * this.pad);
    return [s[0], grow(s[1]), grow(s[2]), s[3]];
  }
  override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const p = this.pad;
    return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], this.value);
  }
  override getConfig() {
    return { ...super.getConfig(), pad: this.pad, value: this.value };
  }
}

tf.serialization.registerClass(HardSwish);
tf.serialization.registerClass(ConstantPad2d);
