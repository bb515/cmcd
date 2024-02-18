"""Calculate the normalising constant using CMCD."""


def main(argv):
    workdir = FLAGS.workdir
    config = FLAGS.config
    jax.default_device = jax.devices()[0]
    # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
    # ... they must be all the same model of device for pmap to work
    num_devices =  int(jax.local_device_count()) if config.training.pmap else 1


if __name__ == "__main__":
    app.run(main)
