package test.utility;

import org.junit.platform.commons.util.StringUtils;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModelName;

public class StringEmbedder {

  public static void main(final String[] args) throws Exception {

    final StringEmbedder embedder = new StringEmbedder();

    final StringEmbedding[] hitEmbeddings = {
      embedder.embed("hitDoc1.txt", "Lucene is a powerful search library."),
      embedder.embed("hitDoc2.txt", "Is Lucinity is not a library?"),
      embedder.embed(
          "hitDoc3.txt", "BM25Similarity is the default search term similarity algorithm.")
    };

    final StringEmbedding[] missEmbeddings = {
      embedder.embed("missDoc1.txt", "Aaaaah - some random text here.")
    };

    final StringEmbedding query1 =
        embedder.embed("query1.txt", "Give me information on the lucine search library");
    query1.toFile();

    final StringEmbedding query2 = embedder.embed("query2.txt", "How to program like Google");
    query2.toFile();

    final StringEmbedding query3 = embedder.embed("query3.txt", "Get rid of mice");
    query3.toFile();

    for (final StringEmbedding stringEmbedding : hitEmbeddings) {
      stringEmbedding.toFile();
    }
    for (final StringEmbedding stringEmbedding : missEmbeddings) {
      stringEmbedding.toFile();
    }
  }

  private final EmbeddingModel embeddingModel;

  public StringEmbedder() {
    this(100);
  }

  public StringEmbedder(final int embeddingDimensions) {
    final String embeddingModelName = OpenAiEmbeddingModelName.TEXT_EMBEDDING_3_SMALL.toString();
    embeddingModel =
        OpenAiEmbeddingModel.builder()
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .modelName(embeddingModelName)
            .dimensions(embeddingDimensions)
            .build();
  }

  public StringEmbedding embed(final String id, final String text) {
    if (StringUtils.isBlank(text)) {
      throw new IllegalArgumentException("No text provided");
    }
    System.out.println("Embedding: " + text);
    final var embedding = embeddingModel.embed(text).content();
    return new StringEmbedding(id, TextSegment.from(text), embedding);
  }
}
