#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
//#include <esp_gap_ble_api.h>

// =========================
// 설정
// =========================

#define SAMPLES_PER_PACKET 20
#define SAMPLE_SIZE        8

#define PACKET_SIZE (SAMPLES_PER_PACKET * SAMPLE_SIZE)

static const char* SERVICE_UUID        = "12345678-1234-1234-1234-1234567890ab";
static const char* CHARACTERISTIC_UUID = "abcd1234-5678-1234-5678-abcdef123456";

// =========================
// BLE
// =========================

BLECharacteristic* pCharacteristic;
bool deviceConnected = false;

// =========================
// 버퍼
// =========================

uint8_t txBuffer[PACKET_SIZE];

volatile uint32_t sampleCounter = 0;

// =========================
// 연결 콜백
// =========================

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;

    Serial.println("BLE Connected");

//    // PHY 2M 요청
//    esp_ble_gap_set_prefered_default_phy(
//      ESP_BLE_GAP_PHY_2M,
//      ESP_BLE_GAP_PHY_2M
//    );
  }

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;

    Serial.println("BLE Disconnected");

    BLEDevice::startAdvertising();
  }
};


#include <Arduino.h>

#include <Adafruit_NeoPixel.h>
Adafruit_NeoPixel strip = Adafruit_NeoPixel(4, 14, NEO_GRB + NEO_KHZ800);

const int periodMicro = 781; // 1280 Hz

#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#define SDA_PIN 34
#define SCL_PIN 35
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29); // 0x29: default, 0x28: AD pin short

int q_int[4] = {0,0,0,10000};
int calib = 0;

const int N_SAMPLE = 20;
const int N_SENSOR = 8;

int trEnablePin = 37;
int powerSwitchPin = 36;
int voltagePin = 13;

int sensorPin[N_SENSOR] = {9, 10, 7, 8, 11, 12, 17, 18};
int emg[N_SAMPLE][N_SENSOR] = {{0,},};
int emg_diff[N_SENSOR] = {0,};
const int led_order[N_SENSOR] = {1, 2, 3, 0};

int getEMG();

void getQuatFromBNO055();

void shutdownOnSwitch();

void setup() {
  Serial.begin(115200);
  pinMode(powerSwitchPin, INPUT_PULLUP);
  pinMode(trEnablePin, OUTPUT);
  //digitalWrite(trEnablePin, LOW);

  strip.begin();
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, i * 2, i * 2, 0);
    }
    strip.show();
    delay(50);
  }

  digitalWrite(trEnablePin, HIGH);

  Wire.begin(SDA_PIN, SCL_PIN);

  while(!bno.begin()) { //compass : 0x09
    for (int i = 0; i < 15; ++i) {
      for (int j = 0; j < 4; ++j) {
        strip.setPixelColor(j, i * 4, 0, i * 4);
      }
      strip.show();
      delay(10);
    }
  }

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 0, i * 4, i * 4);
    }
    strip.show();
    delay(10);
  }

  analogReadResolution(10);

  for (int i = 10; i >= 0; --i) {
    for (int j = 0; j < 4; ++j) {
      strip.setPixelColor(j, 0, i * 4, i * 4);
    }
    strip.show();
    delay(10);
  }

  bno.setExtCrystalUse(true);

  calib = 1; // TEST PURPOSE



  BLEDevice::init("ESP32S3_FAST_BLE");

  // MTU 증가
  BLEDevice::setMTU(247);

  BLEServer* pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );

  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();

  pAdvertising->addServiceUUID(SERVICE_UUID);

  // Connection interval 요청
  pAdvertising->setMinPreferred(0x06); // 7.5ms
  pAdvertising->setMaxPreferred(0x0C); // 15ms

  BLEDevice::startAdvertising();

  Serial.println("BLE Advertising Started");

  delay(100);

}

int getEMG() {
  static int cursor = 0;
  for (int i=0; i<N_SENSOR; ++i) {
    static int tmp = 0;
    tmp = analogRead(sensorPin[i]);
    tmp = 0;
    tmp = analogRead(sensorPin[i]);
    //tmp >= 2;
    //tmp = tmp * 8 / 20; // Previous code on V5.2
    tmp -= 250;
    if (tmp > 255) {
      tmp = 255;
    } else if (tmp < 0) tmp = 0;
    //emg[cursor][i] = tmp;
    txBuffer[cursor*N_SENSOR + i] = tmp;
  }
  char buf[60];
//  snprintf(buf, 60, "%3d %3d %3d %3d %3d %3d %3d %3d",
//    emg[cursor][0], emg[cursor][1], emg[cursor][2], emg[cursor][3],
//    emg[cursor][4], emg[cursor][5], emg[cursor][6], emg[cursor][7]);
//  Serial.println(buf);

//  for (int j = 0; j < 4; ++j) {
//    int val[8] = {0,};
//    if (emg[cursor][j*2] > 0) val[j*2] = abs((emg[cursor][j*2] - 100)/3);
//    if (emg[cursor][j*2+1] > 0) val[j*2+1] = abs((emg[cursor][j*2] - 100)/3);
//    strip.setPixelColor(j, 0, val[j*2], val[j*2+1]);
//  }
//  strip.show();

  cursor++;
  if (cursor >= N_SAMPLE) cursor = 0;
  return cursor;
}

// the loop function runs over and over again forever
void loop() {
  static int i = 0;
  static uint8_t buf[33] = {0,};
  static uint8_t emg_max[4] = {0,};
  static uint8_t emg_min[4] = {0,};
  static uint8_t emg_dif[4] = {0,};
  static uint32_t tick = 0;
  tick = micros();

  // check if the power switch is pressed
  shutdownOnSwitch();

  int pos = getEMG();

  if (pos == 0) {
    //getQuatFromBNO055();
    //// q_int[4] has the value, so 8 bytes should be sent
//    for (int j=0; j<N_SENSOR; ++j) {
//      emg_max[j] = 0;
//      emg_min[j] = 255;
//      for (int i=0; i<N_SAMPLE; ++i) {
//        buf[i*N_SENSOR + j] = (uint8_t)(emg[i][j]);
//        if (buf[i*N_SENSOR + j] > emg_max[j]) emg_max[j] = buf[i*N_SENSOR + j];
//        if (buf[i*N_SENSOR + j] < emg_min[j]) emg_min[j] = buf[i*N_SENSOR + j];
//      }
//      emg_dif[j] = emg_max[j] - emg_min[j];
//    }

//    radio.write(buf, 32);
//    static char sbuf[50];
//    for (int i=0; i<32; i+=4) {
//      snprintf(sbuf, 50, "%d %d %d %d %+4d %+4d %+4d %+4d %d", buf[i], buf[i+1], buf[i+2], buf[i+3], q_int[0]/10, q_int[1]/10, q_int[2]/10, q_int[3]/10, calib);
//      Serial.println(sbuf);
//    }

//    for (int i=0; i<4; ++i) {
//      if (calib > 0) {
//        if (emg_dif[i] < 10) strip.setPixelColor(led_order[i], 0, 0, 0);
//        if (emg_dif[i] < 100) strip.setPixelColor(led_order[i], emg_dif[i]/3, emg_dif[i]/3, emg_dif[i]/3);
//        else if (emg_dif[i] < 150) strip.setPixelColor(led_order[i], emg_dif[i]/5, emg_dif[i]/5, emg_dif[i]/5);
//        else strip.setPixelColor(led_order[i], emg_dif[i]/7, emg_dif[i]/7, emg_dif[i]/7);
//      } else {
//        if (emg_dif[i] < 10) strip.setPixelColor(led_order[i], 0, 0, 0);
//        if (emg_dif[i] < 100) strip.setPixelColor(led_order[i], emg_dif[i]/30, emg_dif[i]/30, emg_dif[i]/30);
//        else if (emg_dif[i] < 150) strip.setPixelColor(led_order[i], emg_dif[i]/40, emg_dif[i]/40, emg_dif[i]/40);
//        else strip.setPixelColor(led_order[i], emg_dif[i]/50, emg_dif[i]/50, emg_dif[i]/50);
//      }
//    }
////    strip.show();

    if (deviceConnected) {
//      for (int j=0; j<N_SENSOR; ++j) {
//        for (int i=0; i<N_SAMPLE; ++i) {
//          txBuffer[i*N_SENSOR + j] = (uint8_t)(emg[i][j]);
//        }
//      }

      pCharacteristic->setValue(txBuffer, PACKET_SIZE);
      pCharacteristic->notify();

      //Serial.print("Notify: ");
      //Serial.println(PACKET_SIZE);
    }


  }
  int delayMicro = periodMicro - micros() + tick;
  if (delayMicro > 0 && delayMicro < periodMicro) {
    delayMicroseconds(delayMicro);
  }
}

void shutdownOnSwitch(){
  int shutdown_count = 33;
  if (digitalRead(powerSwitchPin) == LOW) {
    while (digitalRead(powerSwitchPin) == LOW) {
      for (int j=0; j<4; ++j) {
        strip.setPixelColor(j, shutdown_count, shutdown_count, shutdown_count);
      }
      strip.show();
      delay(50);
      shutdown_count-=2;
      if (shutdown_count < 0) {
        shutdown_count = 0;
        digitalWrite(trEnablePin, LOW);
        for (int j=0; j<4; ++j) {
          //strip.setPixelColor(j, 0, 0, 0);
          digitalWrite(trEnablePin, LOW);
        }
        digitalWrite(trEnablePin, LOW);
        strip.show();
        digitalWrite(trEnablePin, LOW);
        while (1) {
          delay(100);
          digitalWrite(trEnablePin, LOW);
        }
      }
    }
    for (int j=0; j<4; ++j) {
      strip.setPixelColor(j, 0, 0, 0);
    }
    strip.show();
  }
}

void getQuatFromBNO055() {
  /* Get a new sensor event */
  static uint8_t system = 0, gyr = 0, acc = 0, mag = 0;
  static int calibration_loop = 0;
  calibration_loop++;
  if (calibration_loop >= 100) {
    calibration_loop = 0;
    bno.getCalibration(&system, &gyr, &acc, &mag);
    calib = system + mag;
  }
  static uint8_t system_okay = 0;
  if (system > 0) system_okay = 1;

  imu::Quaternion quat = bno.getQuat();
  static float qf[4];
  qf[0] = quat.w();
  qf[1] = quat.x();
  qf[2] = quat.y();
  qf[3] = quat.z();

  for (int i=0; i<4; ++i) {
    q_int[i] = (int)(qf[i] * 10000);
  }
}
