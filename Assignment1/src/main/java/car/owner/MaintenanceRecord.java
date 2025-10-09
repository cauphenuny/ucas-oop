package car.owner;

import java.time.LocalDate;

public record MaintenanceRecord(LocalDate date, String item, double cost, String notes) {}

