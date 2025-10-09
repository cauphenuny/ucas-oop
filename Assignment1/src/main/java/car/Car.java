package car;

public interface Car {
    String getVin();
    String getBrand();
    String getModel();
    int getYear();

    default String getDisplayName() {
        return getBrand() + " " + getModel() + " (" + getYear() + ")";
    }
}
