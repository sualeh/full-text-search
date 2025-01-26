package test.utility;

import org.junit.platform.commons.util.StringUtils;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModelName;
import dev.langchain4j.store.embedding.CosineSimilarity;

public class StringEmbedder {

  public static void main(final String[] args) throws Exception {

    final StringEmbedder embedder = new StringEmbedder();

    final StringEmbedding[] hitEmbeddings = {
      embedder.embed("Lucene is a powerful search library."),
      embedder.embed("Is Lucinity is not a library?"),
      embedder.embed("BM25Similarity is the default search term similarity algorithm.")
    };

    final StringEmbedding[] missEmbeddings = {embedder.embed("Aaaaah - some random text here.")};

    final StringEmbedding query2 = embedder.embed("How to program like Google");
    query2.toFile("query2.txt");

    final StringEmbedding query1 =
        embedder.embed("Give me information on the lucine search library");
    query1.toFile("query1.txt");

    for (int i = 0; i < hitEmbeddings.length; i++) {
      final StringEmbedding stringEmbedding = hitEmbeddings[i];
      stringEmbedding.toFile("hitDoc" + (i + 1) + ".txt");
      final double similarity =
          CosineSimilarity.between(query1.embedding(), stringEmbedding.embedding());
      System.out.printf("%.4f %s%n", similarity, stringEmbedding.text());
    }
    for (int i = 0; i < missEmbeddings.length; i++) {
      final StringEmbedding stringEmbedding = missEmbeddings[i];
      stringEmbedding.toFile("missDoc" + (i + 1) + ".txt");
      final double similarity =
          CosineSimilarity.between(query1.embedding(), stringEmbedding.embedding());
      System.out.printf("%.4f %s%n", similarity, stringEmbedding.text());
    }
  }

  private final EmbeddingModel embeddingModel;

  public StringEmbedder() {
    this(25);
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

  public StringEmbedding embed(final String text) {
    if (StringUtils.isBlank(text)) {
      throw new IllegalArgumentException("No text provided");
    }
    final var embedding = embeddingModel.embed(text).content();
    return new StringEmbedding(TextSegment.from(text), embedding);
  }
}
