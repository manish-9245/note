import org.junit.jupiter.api.BeforeEach;
import org.kie.api.KieServices;
import org.kie.api.builder.KieBuilder;
import org.kie.api.builder.KieFileSystem;
import org.kie.api.builder.Message;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;

public abstract class AbstractDroolsTest {

    protected KieContainer kContainer;
    protected List<String> executionOrder;

    @BeforeEach
    public void setup() throws IOException {
        KieServices kieServices = KieServices.Factory.get();
        KieFileSystem kfs = kieServices.newKieFileSystem();

        // Load execution order from properties file
        Properties props = new Properties();
        props.load(Files.newInputStream(Paths.get("src/test/resources/drools-execution-order.properties")));
        executionOrder = List.of(props.getProperty("execution.order").split(","));

        // Load Drools files in the specified order
        for (String ruleName : executionOrder) {
            String content = new String(Files.readAllBytes(Paths.get("src/main/resources/" + ruleName + ".drl")));
            kfs.write("src/main/resources/" + ruleName + ".drl", content);
        }

        KieBuilder kieBuilder = kieServices.newKieBuilder(kfs).buildAll();
        if (kieBuilder.getResults().hasMessages(Message.Level.ERROR)) {
            throw new RuntimeException("Build Errors:\n" + kieBuilder.getResults().toString());
        }

        kContainer = kieServices.newKieContainer(kieServices.getRepository().getDefaultReleaseId());
    }

    protected <T> T runRules(List<String> rulesToRun, Object... facts) {
        KieSession kSession = kContainer.newKieSession();
        try {
            for (Object fact : facts) {
                kSession.insert(fact);
            }

            for (String rule : rulesToRun) {
                kSession.getAgenda().getAgendaGroup(rule).setFocus();
            }

            kSession.fireAllRules();

            return getResult(kSession);
        } finally {
            kSession.dispose();
        }
    }

    protected abstract <T> T getResult(KieSession kSession);
}

// Example of a concrete test class
class ConcreteDroolsTest extends AbstractDroolsTest {

    @Test
    void testSpecificRules() {
        List<String> rulesToRun = List.of("Rule1", "Rule2", "Rule3");
        MyResult result = runRules(rulesToRun, new Fact("fact1"), new Fact("fact2"));
        
        // Assert on the result
        assertEquals(expectedValue, result.getValue());
    }

    @Override
    protected <T> T getResult(KieSession kSession) {
        // Extract and return the result from the KieSession
        // This will depend on how your rules produce results
        return (T) kSession.getGlobal("result");
    }

    // Define your Fact and MyResult classes here
}
