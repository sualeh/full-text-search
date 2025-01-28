package test.utility;

import org.junit.platform.commons.util.StringUtils;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModelName;

public class TextEmbedder {

  public static void main(final String[] args) throws Exception {

    final TextEmbedder embedder = new TextEmbedder();

    final TextEmbedding[] hitEmbeddings = {
      embedder.embed("hitDoc1.txt", "Lucene is a powerful search library."),
      embedder.embed(
          "hitDoc2.txt", "BM25Similarity is the default search term similarity algorithm."),
      embedder.embed("hitDoc3.txt", "Is Lucinity is not a library?"),
    };

    final TextEmbedding[] missEmbeddings = {
      embedder.embed("missDoc1.txt", "Aaaaah - some random text here."),
    };

    final TextEmbedding query1 =
        embedder.embed("query1.txt", "Give me information on the lucine search library");
    query1.toFile();

    final TextEmbedding query2 = embedder.embed("query2.txt", "How to program like Google");
    query2.toFile();

    final TextEmbedding query3 = embedder.embed("query3.txt", "Get rid of mice");
    query3.toFile();

    for (final TextEmbedding stringEmbedding : hitEmbeddings) {
      stringEmbedding.toFile();
    }
    for (final TextEmbedding stringEmbedding : missEmbeddings) {
      stringEmbedding.toFile();
    }
  }

  private final EmbeddingModel embeddingModel;

  public TextEmbedder() {
    final int embeddingDimensions = 50;
    final String embeddingModelName = OpenAiEmbeddingModelName.TEXT_EMBEDDING_3_SMALL.toString();
    final String apiKey = System.getenv("OPENAI_API_KEY");
    embeddingModel =
        OpenAiEmbeddingModel.builder()
            .apiKey(apiKey)
            .modelName(embeddingModelName)
            .dimensions(embeddingDimensions)
            .build();
  }

  public TextEmbedding embed(final String id, final String text) {
    if (StringUtils.isBlank(text)) {
      throw new IllegalArgumentException("No text provided");
    }
    System.out.println("Embedding: " + text);
    final var embedding = embeddingModel.embed(text).content();
    return new TextEmbedding(id, TextSegment.from(text), embedding);
  }
}
