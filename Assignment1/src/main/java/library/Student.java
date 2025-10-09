package library;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * 学生实体：可借阅和归还书籍。
 */
public class Student {
    private final String studentId;                              // 学号唯一标识
    private String name;                                         // 姓名
    private int maxBorrowLimit = 5;                              // 最大可借数量（可配置）
    private final List<Book> borrowedBooks = new ArrayList<>();  // 当前借阅列表

    public Student(String studentId, String name) {
        this.studentId = Objects.requireNonNull(studentId, "studentId不能为空");
        this.name = Objects.requireNonNull(name, "name不能为空");
    }

    /**
     * 学生尝试借书：调用图书馆服务（管理员）来完成，这里仅做前置校验。
     */
    void addBorrowedBook(Book book) {
        if (borrowedBooks.size() >= maxBorrowLimit) {
            throw new IllegalStateException("超出最大借阅数量: " + maxBorrowLimit);
        }
        borrowedBooks.add(book);
    }

    /**
     * 学生归还书籍后从列表移除。
     */
    void removeBorrowedBook(Book book) { borrowedBooks.remove(book); }

    public List<Book> getBorrowedBooks() { return Collections.unmodifiableList(borrowedBooks); }

    public boolean hasBorrowed(String isbn) {
        return borrowedBooks.stream().anyMatch(b -> b.getIsbn().equals(isbn));
    }

    public boolean hasOverdueBooks(LocalDate today) {
        return borrowedBooks.stream().anyMatch(b -> b.isOverDue(today));
    }

    public String getStudentId() { return studentId; }
    public String getName() { return name; }
    public void setName(String name) { this.name = Objects.requireNonNull(name); }
    public int getMaxBorrowLimit() { return maxBorrowLimit; }
    public void setMaxBorrowLimit(int maxBorrowLimit) {
        if (maxBorrowLimit <= 0) throw new IllegalArgumentException("maxBorrowLimit 必须 > 0");
        this.maxBorrowLimit = maxBorrowLimit;
    }

    @Override
    public String toString() {
        return "Student{"
            + "studentId='" + studentId + '\'' + ", name='" + name + '\'' +
            ", borrowedCount=" + borrowedBooks.size() + '}';
    }
}
