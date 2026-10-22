import {
  SERIAL_BAUD,
  SERIAL_BUFFER_LIMIT,
} from '../core/constants.js';

const MESSAGE = Object.freeze({
  TOUCH: 'TOUCH',
  RELEASE: 'RELEASE',
});

export function parseSerialLine(rawLine) {
  const line = rawLine.trim();

  if (line === MESSAGE.TOUCH) {
    return { type: 'touch' };
  }

  if (line === MESSAGE.RELEASE) {
    return { type: 'release' };
  }

  if (line.startsWith('READY:')) {
    return {
      type: 'ready',
      device: line.slice('READY:'.length),
    };
  }

  if (line.startsWith('ERROR:')) {
    return {
      type: 'device-error',
      message: line.slice('ERROR:'.length),
    };
  }

  return null;
}

export class SerialLink {
  constructor() {
    this.port = null;
    this.reader = null;
    this.reading = false;
    this.connecting = false;
    this.listeners = new Set();

    this.handleDisconnect =
      this.handleDisconnect.bind(this);

    globalThis.navigator?.serial?.addEventListener(
      'disconnect',
      this.handleDisconnect
    );
  }

  get supported() {
    return Boolean(globalThis.navigator?.serial);
  }

  get connected() {
    return Boolean(
      this.port?.readable && this.reading
    );
  }

  subscribe(listener) {
    this.listeners.add(listener);

    return () => {
      this.listeners.delete(listener);
    };
  }

  async connect() {
    if (!this.supported) {
      this.emit({ type: 'unsupported' });
      return false;
    }

    if (this.connected || this.connecting) {
      return true;
    }

    this.connecting = true;
    this.emit({ type: 'connecting' });

    try {
      const port =
        await navigator.serial.requestPort();

      await port.open({
        baudRate: SERIAL_BAUD,
      });

      this.port = port;
      this.reading = true;

      this.emit({ type: 'connected' });
      void this.read(port);

      return true;
    } catch (error) {
      const cancelled =
        error?.name === 'NotFoundError';

      this.emit({
        type: cancelled
          ? 'cancelled'
          : 'error',
        error,
      });

      return false;
    } finally {
      this.connecting = false;
    }
  }

  async read(port) {
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (this.reading && port.readable) {
        this.reader =
          port.readable.getReader();

        try {
          while (this.reading) {
            const { value, done } =
              await this.reader.read();

            if (done) break;

            buffer += decoder.decode(
              value,
              { stream: true }
            );

            if (
              buffer.length
              > SERIAL_BUFFER_LIMIT
            ) {
              buffer = '';

              this.emit({
                type: 'buffer-error',
              });

              continue;
            }

            const lines =
              buffer.split(/\r?\n/);

            buffer = lines.pop() ?? '';

            for (const line of lines) {
              const message =
                parseSerialLine(line);

              if (message) {
                this.emit(message);
              }
            }
          }
        } finally {
          this.reader.releaseLock();
          this.reader = null;
        }
      }
    } catch (error) {
      if (this.reading) {
        this.emit({
          type: 'error',
          error,
        });
      }
    }
  }

  async disconnect(silent = false) {
    this.reading = false;

    if (this.reader) {
      await this.reader
        .cancel()
        .catch(() => {});
    }

    if (this.port) {
      await this.port
        .close()
        .catch(() => {});
    }

    this.reader = null;
    this.port = null;

    if (!silent) {
      this.emit({
        type: 'disconnected',
      });
    }
  }

  handleDisconnect(event) {
    const removedPort =
      event.port ?? event.target;

    if (removedPort !== this.port) return;

    void this.disconnect(true).then(() => {
      this.emit({
        type: 'disconnected',
      });
    });
  }

  emit(event) {
    for (const listener of this.listeners) {
      listener(event);
    }
  }
}