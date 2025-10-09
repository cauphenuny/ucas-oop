package library;

import java.time.LocalDate;
import java.util.*;

/**
 * 管理员：负责管理书目及借还流程。
 */
public class Librarian {
    private final String employeeId;  // 工号
    private String name;

    // 图书馆的藏书目录（可简单用 Map 存储）
    private final Map<String, Book> catalogue = new HashMap<>();  // key: isbn

    public Librarian(String employeeId, String name) {
        this.employeeId = Objects.requireNonNull(employeeId, "employeeId不能为空");
        this.name = Objects.requireNonNull(name, "name不能为空");
    }

    /**
     * 新增图书到目录。
     */
    public void addBook(Book book) {
        if (catalogue.containsKey(book.getIsbn())) {
            throw new IllegalArgumentException("ISBN 已存在: " + book.getIsbn());
        }
        catalogue.put(book.getIsbn(), book);
    }

    /**
     * 删除图书（需未被借出）。
     */
    public void removeBook(String isbn) {
        Book b = getBook(isbn);
        if (b.isBorrowed()) {
            throw new IllegalStateException("书籍仍被借出，不能删除");
        }
        catalogue.remove(isbn);
    }

    public Book getBook(String isbn) {
        Book b = catalogue.get(isbn);
        if (b == null) throw new NoSuchElementException("未找到图书: " + isbn);
        return b;
    }

    /**
     * 借书流程：
     * 1. 校验学生未超限、无逾期
     * 2. 校验书籍存在且未借出
     * 3. 更新书籍状态 + 学生借阅列表
     */
    public void borrowBook(Student student, String isbn, int borrowDays) {
        Objects.requireNonNull(student, "student不能为空");
        if (borrowDays <= 0) throw new IllegalArgumentException("borrowDays 必须 > 0");
        if (student.hasOverdueBooks(LocalDate.now())) {
            throw new IllegalStateException("存在逾期书籍，不能继续借书");
        }
        Book book = getBook(isbn);
        if (book.isBorrowed()) {
            throw new IllegalStateException("书籍已被借出");
        }
        student.addBorrowedBook(book);
        LocalDate today = LocalDate.now();
        book.markBorrowed(student.getStudentId(), today, today.plusDays(borrowDays));
    }

    /**
     * 还书流程：
     * 1. 校验书籍确实由该学生借出
     * 2. 更新书籍状态 + 学生列表
     */
    public void returnBook(Student student, String isbn) {
        Objects.requireNonNull(student, "student不能为空");
        Book book = getBook(isbn);
        if (!book.isBorrowed() || !student.hasBorrowed(isbn)) {
            throw new IllegalStateException("该学生未借阅此书");
        }
        book.markReturned();
        student.removeBorrowedBook(book);
    }

    /**
     * 查询逾期书籍。
     */
    public List<Book> listOverdueBooks() {
        LocalDate today = LocalDate.now();
        List<Book> result = new ArrayList<>();
        for (Book b : catalogue.values()) {
            if (b.isOverDue(today)) result.add(b);
        }
        return result;
    }

    public Collection<Book> listAllBooks() {
        return Collections.unmodifiableCollection(catalogue.values());
    }

    public String getEmployeeId() { return employeeId; }
    public String getName() { return name; }
    public void setName(String name) { this.name = Objects.requireNonNull(name); }

    @Override
    public String toString() {
        return "Librarian{"
            + "employeeId='" + employeeId + '\'' + ", name='" + name + '\'' +
            ", bookCount=" + catalogue.size() + '}';
    }
}
