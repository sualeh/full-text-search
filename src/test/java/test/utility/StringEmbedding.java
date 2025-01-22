package test.utility;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import static java.util.Objects.requireNonNull;
import dev.langchain4j.data.embedding.Embedding;

public record StringEmbedding(String text, Embedding embedding) {

  public StringEmbedding {
    requireNonNull(text, "No text provided");
    requireNonNull(embedding, "No embedding provided");
  }

  public static StringEmbedding readFromFile(final String filename) throws IOException {
    final List<String> lines = Files.readAllLines(Paths.get(filename));
    if (lines.size() != 2) {
      throw new IOException("Expected 2 lines");
    }
    final var text = lines.get(0);
    final float[] array;
    final var line = lines.get(1);
    if (line != null) {
      final String[] tokens = line.split(",");
      array = new float[tokens.length];
      for (int i = 0; i < tokens.length; i++) {
        array[i] = Float.parseFloat(tokens[i]);
      }
    } else {
      array = new float[0];
    }
    return new StringEmbedding(text, Embedding.from(array));
  }

  public void writeToFile(final String filename) throws IOException {
    try (final PrintWriter writer = new PrintWriter(filename)) {
      writer.println(text());
      final var vector = embedding.vector();
      for (int i = 0; i < vector.length; i++) {
        writer.write(Float.toString(vector[i]));
        if (i < vector.length - 1) {
          writer.write(",");
        }
      }
      writer.println();
    }
  }
}
