package car.owner;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 车主视角：所有权、车牌、里程、保养、保险。
 */
public class OwnerCar implements car.Car {
    private final String vin;
    private final String brand;
    private final String model;
    private final int year;

    private String plateNumber;                // 车牌
    private String ownerId;                    // 证件/学号等
    private String ownerName;
    private long odometerKm;                   // 里程（公里）
    private final List<MaintenanceRecord> maintenanceRecords = new ArrayList<>();

    private String insuranceCompany;
    private LocalDate insuranceExpireDate;

    public OwnerCar(String vin, String brand, String model, int year, String plateNumber, String ownerId, String ownerName) {
        this.vin = Objects.requireNonNull(vin);
        this.brand = Objects.requireNonNull(brand);
        this.model = Objects.requireNonNull(model);
        this.year = year;
        this.plateNumber = Objects.requireNonNull(plateNumber);
        this.ownerId = Objects.requireNonNull(ownerId);
        this.ownerName = Objects.requireNonNull(ownerName);
    }

    // 业务操作
    public void drive(long km) {
        if (km > 0) this.odometerKm += km;
    }

    public void addMaintenance(MaintenanceRecord record) {
        this.maintenanceRecords.add(Objects.requireNonNull(record));
    }

    public void insure(String company, LocalDate expireDate) {
        this.insuranceCompany = company;
        this.insuranceExpireDate = expireDate;
    }

    public boolean isInsured(LocalDate onDate) {
        return insuranceExpireDate != null && !insuranceExpireDate.isBefore(onDate);
    }

    public void transferOwnership(String newOwnerId, String newOwnerName) {
        this.ownerId = Objects.requireNonNull(newOwnerId);
        this.ownerName = Objects.requireNonNull(newOwnerName);
    }

    // getters
    @Override public String getVin() { return vin; }
    @Override public String getBrand() { return brand; }
    @Override public String getModel() { return model; }
    @Override public int getYear() { return year; }

    public String getPlateNumber() { return plateNumber; }
    public String getOwnerId() { return ownerId; }
    public String getOwnerName() { return ownerName; }
    public long getOdometerKm() { return odometerKm; }
    public List<MaintenanceRecord> getMaintenanceRecords() { return new ArrayList<>(maintenanceRecords); }
    public String getInsuranceCompany() { return insuranceCompany; }
    public LocalDate getInsuranceExpireDate() { return insuranceExpireDate; }

    public void setPlateNumber(String plateNumber) { this.plateNumber = plateNumber; }

    @Override
    public String toString() {
        return "OwnerCar{" + vin + ", " + getDisplayName() + ", plate=" + plateNumber + ", owner=" + ownerName + "}";
    }
}
