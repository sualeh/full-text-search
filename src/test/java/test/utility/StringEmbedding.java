package test.utility;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import static java.util.Objects.requireNonNull;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

public record StringEmbedding(String id, TextSegment text, Embedding embedding) {

  public StringEmbedding {
    requireNonNull(text, "No text provided");
    requireNonNull(embedding, "No embedding provided");
  }

  public static StringEmbedding fromResource(final String resourceName) {
    try {
      final URL resource = StringEmbedding.class.getResource("/" + resourceName);
      final Path resourcePath = Paths.get(resource.toURI());
      final List<String> lines = Files.readAllLines(resourcePath);
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
      final Metadata metadata = metadataName(resourceName);
      return new StringEmbedding(
          resourceName, TextSegment.from(text, metadata), Embedding.from(array));
    } catch (final Exception e) {
      return new StringEmbedding(
          resourceName, TextSegment.from(e.getMessage()), Embedding.from(new float[0]));
    }
  }

  private static Metadata metadataName(final String name) {
    final Metadata metadata = new Metadata();
    metadata.put("name", name);
    return metadata;
  }

  public void toFile() throws IOException {
    try (final PrintWriter writer = new PrintWriter(id())) {
      writer.println(text().text());
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
