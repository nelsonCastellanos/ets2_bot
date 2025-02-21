from internal.infrastructure.data_logger import DataLogger
from internal.infrastructure.window_capture_quartz import QuartzWindowCapture
from internal.domain.simulator.gps_processor import GPSImageProcessorImpl


def main():
    window_title = "Euro Truck Simulator"
    quartz_capture = QuartzWindowCapture(window_title)
    gps_processor = GPSImageProcessorImpl()

    logger = DataLogger(screen_capture=quartz_capture, gps_processor=gps_processor)
    logger.start_key_listener()

    print("Iniciando captura de datos para entrenamiento. Presiona Ctrl+C para detener.")
    try:
        logger.capture_and_log(num_frames=200, delay=0.1)
    except KeyboardInterrupt:
        print("Interrupción manual.")
    finally:
        logger.stop_key_listener()
        logger.close()


if __name__ == '__main__':
    main()
