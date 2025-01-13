#include <DHTesp.h>

DHTesp dht; // Create an instance of DHTesp

// Define the GPIO pin for the DHT sensor
const int DHT_PIN = 15;

void setup() {
  Serial.begin(115200);
  dht.setup(DHT_PIN, DHTesp::DHT22); // Initialize the DHT sensor (use DHTesp::DHT11 for DHT11)
  Serial.println("DHT Sensor Simulation Started");
}

void loop() {
  // Read temperature and humidity
  TempAndHumidity data = dht.getTempAndHumidity();

  // Check for errors
  if (isnan(data.temperature) || isnan(data.humidity)) {
    Serial.println("Failed to read from DHT sensor!");
  } else {
    // Print temperature and humidity
    Serial.print("Temperature: ");
    Serial.print(data.temperature);
    Serial.println("°C");

    Serial.print("Humidity: ");
    Serial.print(data.humidity);
    Serial.println("%");
  }

  delay(5000); // Wait for 2 seconds before the next reading
}
