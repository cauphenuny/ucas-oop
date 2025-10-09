package car.manager;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 交通管理机构视角：注册状态、年检、违章、罚款。
 */
public class ManagerCar implements car.Car {
    private final String vin;
    private final String brand;
    private final String model;
    private final int year;

    private String registrationId;                 // 注册号
    private String plateNumber;                    // 车牌号
    private RegistrationStatus status = RegistrationStatus.UNREGISTERED;
    private LocalDate inspectionExpireDate;        // 年检有效期
    private final List<Violation> violations = new ArrayList<>();

    public ManagerCar(String vin, String brand, String model, int year) {
        this.vin = Objects.requireNonNull(vin);
        this.brand = Objects.requireNonNull(brand);
        this.model = Objects.requireNonNull(model);
        this.year = year;
    }

    // 业务操作
    public void register(String registrationId, String plateNumber, LocalDate inspectionExpireDate) {
        this.registrationId = Objects.requireNonNull(registrationId);
        this.plateNumber = Objects.requireNonNull(plateNumber);
        this.inspectionExpireDate = inspectionExpireDate;
        this.status = RegistrationStatus.ACTIVE;
    }

    public void suspend() {
        this.status = RegistrationStatus.SUSPENDED;
    }

    public void reinstate() {
        this.status = RegistrationStatus.ACTIVE;
    }

    public void addViolation(String code, int points, double fine, LocalDate date) {
        this.violations.add(new Violation(code, points, fine, date));
    }

    public double calculateTotalFine() {
        return violations.stream().mapToDouble(Violation::fine).sum();
    }

    public int calculateTotalPoints() {
        return violations.stream().mapToInt(Violation::points).sum();
    }

    public boolean isInspectionValid(LocalDate onDate) {
        return inspectionExpireDate != null && !inspectionExpireDate.isBefore(onDate);
    }

    // getters
    @Override public String getVin() { return vin; }
    @Override public String getBrand() { return brand; }
    @Override public String getModel() { return model; }
    @Override public int getYear() { return year; }

    public String getRegistrationId() { return registrationId; }
    public String getPlateNumber() { return plateNumber; }
    public RegistrationStatus getStatus() { return status; }
    public LocalDate getInspectionExpireDate() { return inspectionExpireDate; }
    public List<Violation> getViolations() { return new ArrayList<>(violations); }

    @Override
    public String toString() {
        return "AuthorityCar{" + vin + ", " + getDisplayName() + ", status=" + status + "}";
    }
}
