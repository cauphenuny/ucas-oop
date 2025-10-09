package car.manager;

import java.time.LocalDate;

public record Violation(String code, int points, double fine, LocalDate date) {}