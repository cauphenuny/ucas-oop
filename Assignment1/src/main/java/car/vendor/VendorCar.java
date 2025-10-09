package car.vendor;

import java.time.LocalDate;
import java.util.Objects;

public class VendorCar implements car.Car {
    private final String vin;
    private final String brand;
    private final String model;
    private final int year;

    private final LocalDate productionDate;
    private int warrantyYears;             // 质保年限
    private boolean recalled;              // 是否处于召回中
    private String recallNotes;            // 召回说明
    private String softwareVersion;        // 车载软件版本

    public VendorCar(String vin, String brand, String model, int year, LocalDate productionDate, int warrantyYears) {
        this.vin = Objects.requireNonNull(vin);
        this.brand = Objects.requireNonNull(brand);
        this.model = Objects.requireNonNull(model);
        this.year = year;
        this.productionDate = Objects.requireNonNull(productionDate);
        this.warrantyYears = warrantyYears;
        this.softwareVersion = "1.0.0";
    }

    // 业务操作
    public void startRecall(String notes) {
        this.recalled = true;
        this.recallNotes = notes;
    }

    public void endRecall() {
        this.recalled = false;
        this.recallNotes = null;
    }

    public boolean isUnderWarranty(LocalDate onDate) {
        return productionDate.plusYears(warrantyYears).isAfter(onDate);
    }

    public void updateSoftware(String newVersion) {
        this.softwareVersion = newVersion;
    }

    // getters
    @Override public String getVin() { return vin; }
    @Override public String getBrand() { return brand; }
    @Override public String getModel() { return model; }
    @Override public int getYear() { return year; }

    public LocalDate getProductionDate() { return productionDate; }
    public int getWarrantyYears() { return warrantyYears; }
    public boolean isRecalled() { return recalled; }
    public String getRecallNotes() { return recallNotes; }
    public String getSoftwareVersion() { return softwareVersion; }

    public void setWarrantyYears(int warrantyYears) { this.warrantyYears = warrantyYears; }

    @Override
    public String toString() {
        return "ManufacturerCar{" + vin + ", " + getDisplayName() + ", recalled=" + recalled + "}";
    }
}
