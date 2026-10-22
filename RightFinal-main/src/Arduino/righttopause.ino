#include <Wire.h>
#include <Adafruit_MPR121.h>
/** Hardware and communication settings. */
const uint32_t SERIAL_BAUD = 115200;
const uint8_t MPR121_ADDRESS = 0x5A;
const uint8_t TOUCH_ELECTRODE = 0;

/** Sensor sensitivity. */
const uint8_t TOUCH_THRESHOLD = 6;
const uint8_t RELEASE_THRESHOLD = 3;

/** Ignore very short electrical fluctuations. */
const uint16_t DEBOUNCE_MS = 35;

Adafruit_MPR121 sensor;

bool lastReading = false;
bool stableTouch = false;
unsigned long changedAt = 0;

/**
 * Read the current state of electrode 0.
 */
bool readTouch() {
  const uint16_t electrodes =
    sensor.touched();

  return electrodes
    & (1U << TOUCH_ELECTRODE);
}

/**
 * Send one message when contact changes.
 */
void reportTouch(bool touching) {
  if (touching) {
    Serial.println(F("TOUCH"));
  } else {
    Serial.println(F("RELEASE"));
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);

  if (!sensor.begin(MPR121_ADDRESS)) {
    Serial.println(
      F("ERROR:MPR121_NOT_FOUND")
    );

    while (true) {
      delay(1000);
    }
  }
  sensor.setThresholds(
    TOUCH_THRESHOLD,
    RELEASE_THRESHOLD
  );
  lastReading = readTouch();
  stableTouch = lastReading;

  Serial.println(F("READY:MPR121"));
  reportTouch(stableTouch);
}

void loop() {
  const bool reading = readTouch();
  const unsigned long now = millis();
  // Print the raw state of electrode 0 for testing.
static unsigned long lastDebugAt = 0;

if (now - lastDebugAt >= 200) {
  lastDebugAt = now;

  Serial.print(F("E0:"));
  Serial.print(
    sensor.touched() & 1U
      ? 1
      : 0
  );

  Serial.print(F(" FILTERED:"));
  Serial.print(sensor.filteredData(0));

  Serial.print(F(" BASELINE:"));
  Serial.println(sensor.baselineData(0));
}

  // Restart debounce timing when raw input changes.
  if (reading != lastReading) {
    lastReading = reading;
    changedAt = now;
  }

  // Accept the change only after it remains stable.
  if (
    reading != stableTouch
    && now - changedAt >= DEBOUNCE_MS
  ) {
    stableTouch = reading;
    reportTouch(stableTouch);
  }

  delay(5);
}

//Read the sensor
// Check ELE0
// Software debounce
// Only send serial messages when the state changes