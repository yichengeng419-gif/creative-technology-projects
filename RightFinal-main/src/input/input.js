// Names of the input sources that can own a hold. 
const SOURCE = Object.freeze({
  POINTER: 'pointer',
  KEYBOARD: 'keyboard',
  SERIAL: 'serial',
});
// Elements whose clicks must not become specimen contact. 
const INTERACTIVE =
  'button,input,select,textarea,a,details,summary';

export class Input {
  constructor(target = window) {
    this.target = target;

    // Only one input source can own a hold.
    this.owner = null;
    this.touching = false;
    this.pointerEnabled = true;
    this.position = [0, 0];

    this.listeners = new Set();

    this.handleDown =
      this.handleDown.bind(this);

    this.handleMove =
      this.handleMove.bind(this);

    this.handleUp =
      this.handleUp.bind(this);

    this.handleKeyDown =
      this.handleKeyDown.bind(this);

    this.handleKeyUp =
      this.handleKeyUp.bind(this);

    this.handleBlur =
      this.handleBlur.bind(this);
  }
  connect() {
    this.target.addEventListener(
      'pointerdown',
      this.handleDown
    );

    this.target.addEventListener(
      'pointermove',
      this.handleMove
    );

    window.addEventListener(
      'pointerup',
      this.handleUp
    );

    window.addEventListener(
      'pointercancel',
      this.handleUp
    );

    window.addEventListener(
      'keydown',
      this.handleKeyDown
    );

    window.addEventListener(
      'keyup',
      this.handleKeyUp
    );

    window.addEventListener(
      'blur',
      this.handleBlur
    );
  }

  // Convert browser coordinates to WebGL coordinates.
  handleMove(event) {
    const rect =
      this.target.getBoundingClientRect();

    const x =
      (event.clientX - rect.left)
      / rect.width;

    const y =
      (event.clientY - rect.top)
      / rect.height;

    this.position[0] = x * 2 - 1;
    this.position[1] = 1 - y * 2;
  }

  handleDown(event) {
    if (!this.pointerEnabled) return;
    if (event.button !== 0) return;

    if (
      event.target
        ?.closest
        ?.(INTERACTIVE)
    ) {
      return;
    }

    this.handleMove(event);

    this.begin(SOURCE.POINTER);
  }

  handleUp() {
    this.end(SOURCE.POINTER);
  }

  handleKeyDown(event) {
    if (!this.pointerEnabled) return;
    if (event.repeat) return;
    if (event.code !== 'Space') return;

    if (
      event.target
        ?.closest
        ?.(INTERACTIVE)
    ) {
      return;
    }

    if (this.begin(SOURCE.KEYBOARD)) {
      event.preventDefault();
    }
  }

  handleKeyUp(event) {
    if (event.code !== 'Space') return;

    if (this.end(SOURCE.KEYBOARD)) {
      event.preventDefault();
    }
  }

  handleBlur() {
    if (
      this.owner === SOURCE.POINTER
      || this.owner === SOURCE.KEYBOARD
    ) {
      this.end(this.owner, true);
    }
  }

  /**
   * Give one source ownership of the current hold.
   */
  begin(source) {
    if (this.owner !== null) return false;

    this.owner = source;
    this.touching = true;

    this.emit({
      touching: true,
      source,
      cancelled: false,
    });

    return true;
  }

  /**
   * End a hold only when the same source owns it.
   */
  end(source, cancelled = false) {
    if (this.owner !== source) return false;

    this.owner = null;
    this.touching = false;

    this.emit({
      touching: false,
      source,
      cancelled,
    });

    return true;
  }

  // Entry point used by external hardware such as MPR121.

  setTouching(
    value,
    source = SOURCE.SERIAL
  ) {
    if (value) {
      return this.begin(source);
    }

    return this.end(source);
  }

  //Disable mouse and keyboard while serial owns input.
  setPointerEnabled(enabled) {
    this.pointerEnabled =
      Boolean(enabled);

    if (
      !this.pointerEnabled
      && (
        this.owner === SOURCE.POINTER
        || this.owner === SOURCE.KEYBOARD
      )
    ) {
      this.end(this.owner, true);
    }
  }

  subscribe(listener) {
    this.listeners.add(listener);

    return () => {
      this.listeners.delete(listener);
    };
  }

  emit(message) {
    const event = {
      ...message,
      time: performance.now(),
      position: [...this.position],
    };

    for (const listener of this.listeners) {
      listener(event);
    }
  }

  disconnect() {
    if (this.owner !== null) {
      this.end(this.owner, true);
    }

    this.target.removeEventListener(
      'pointerdown',
      this.handleDown
    );

    this.target.removeEventListener(
      'pointermove',
      this.handleMove
    );

    window.removeEventListener(
      'pointerup',
      this.handleUp
    );

    window.removeEventListener(
      'pointercancel',
      this.handleUp
    );

    window.removeEventListener(
      'keydown',
      this.handleKeyDown
    );

    window.removeEventListener(
      'keyup',
      this.handleKeyUp
    );

    window.removeEventListener(
      'blur',
      this.handleBlur
    );
  }
}