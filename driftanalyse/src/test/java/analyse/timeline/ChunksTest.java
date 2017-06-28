package analyse.timeline;

import global.DriftMeasurement;
import models.Model;
import models.frequency.FrequencyMaps;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import result.timeline.TimelineListener;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Created by LoongKuan on 25/06/2017.
 */
public class ChunksTest {
    private static ArrayList<String> values = new ArrayList<>(Arrays.asList("0", "1", "2", "3", "4"));
    private static Attribute feature = new Attribute("feature", values);
    private static Attribute group = new Attribute("group", values);
    private static ArrayList<Attribute> attributes = new ArrayList<>(Arrays.asList(feature, group));
    private static Instances instances = new Instances("test", attributes, 0);

    private static boolean setupDone = false;

    private static Instances generateData(String feature, String group, int amount) {
        Instances newInstances = new Instances(instances, amount);
        double[] values = new double[]{Double.parseDouble(feature), Double.parseDouble(group)};
        for (int i = 0; i < amount; i++) {
            newInstances.add(new DenseInstance(1.0, values));
        }
        newInstances.setClassIndex(1);
        return newInstances;
    }

    private static void addInstances(Chunks chunks, String feature, String group, int amount) {
        Instances instances = generateData(feature, group, amount);
        for (Instance instance : instances) {
            chunks.addInstance(instance);
        }
    }

    @BeforeClass
    public static void setup() {
        if (setupDone) {
            return;
        }
        instances.setClassIndex(1);
        setupDone = true;
    }

    @Test
    public void testGrowPreviousModel() {
        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 2, true, model);

        addInstances(chunks, "0", "0", 2);
        assertEquals(2, chunks.getPreviousModel().getAllInstances().size());
        assertEquals(2, chunks.getCurrentIndex());
        assertEquals(0, chunks.getCurrentModel().getAllInstances().size());

        addInstances(chunks, "1", "1", 1);
        assertEquals(3, chunks.getPreviousModel().getAllInstances().size());
        assertEquals(3, chunks.getCurrentIndex());
        assertEquals(0, chunks.getCurrentModel().getAllInstances().size());
    }

    @Test
    public void testGrowCurrentModel() {
        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 2, true, model);
        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "1", "1", 1);

        addInstances(chunks, "2", "2", 1);
        assertEquals(3, chunks.getPreviousModel().getAllInstances().size());
        assertEquals(3, chunks.getCurrentIndex());
        assertEquals(1, chunks.getCurrentModel().getAllInstances().size());

        addInstances(chunks, "3", "3", 2);
        assertEquals(3, chunks.getPreviousModel().getAllInstances().size());
        assertEquals(3, chunks.getCurrentIndex());
        assertEquals(3, chunks.getCurrentModel().getAllInstances().size());
    }

    @Test
    public void testMoveModels() {
        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 2, true, model);
        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "1", "1", 1);
        addInstances(chunks, "2", "2", 1);
        addInstances(chunks, "3", "3", 2);

        addInstances(chunks, "4", "4", 2);
        assertEquals(2, chunks.getPreviousModel().getAllInstances().size());
        assertEquals(4, chunks.getCurrentIndex());
        assertEquals(4, chunks.getCurrentModel().getAllInstances().size());
    }

    @Test
    public void testNoIncrement() {
        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 2, false, model);
        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "1", "1", 1);
        addInstances(chunks, "2", "2", 1);
        addInstances(chunks, "3", "3", 2);

        addInstances(chunks, "4", "4", 2);
        assertEquals(3, chunks.getPreviousModel().getAllInstances().size());
        assertEquals(6, chunks.getCurrentIndex());
        assertEquals(2, chunks.getCurrentModel().getAllInstances().size());
    }

    @Test
    public void testCallBackCovariate() {
        // Create Listeners
        TimelineListener listener = mock(TimelineListener.class);
        when(listener.getMeasurementType()).thenReturn(DriftMeasurement.COVARIATE);

        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 1, true, model);
        chunks.addListener(listener);

        addInstances(chunks, "0", "3", 2);
        addInstances(chunks, "3", "3", 2);

        chunks.addDistance(chunks.currentIndex);
        verify(listener).returnDriftPointMagnitude(chunks.currentIndex, new double[]{1.0});
    }

    @Test
    public void testCallBackClass() {
        // Create Listeners
        TimelineListener listener = mock(TimelineListener.class);
        when(listener.getMeasurementType()).thenReturn(DriftMeasurement.CLASS);

        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 1, true, model);
        chunks.addListener(listener);

        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "3", "3", 2);

        chunks.addDistance(chunks.currentIndex);
        verify(listener).returnDriftPointMagnitude(chunks.currentIndex, new double[]{1.0});
    }

    @Test
    public void testCallBackLikelihood() {
        // Create Listeners
        TimelineListener listener = mock(TimelineListener.class);
        when(listener.getMeasurementType()).thenReturn(DriftMeasurement.LIKELIHOOD);

        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 1, true, model);
        chunks.addListener(listener);

        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "3", "0", 2);
        chunks.addDistance(chunks.currentIndex);
        verify(listener).returnDriftPointMagnitude(chunks.currentIndex, new double[]{1.0});
    }

    @Test
    public void testCallBackPosterior() {
        // Create Listeners
        TimelineListener listener = mock(TimelineListener.class);
        when(listener.getMeasurementType()).thenReturn(DriftMeasurement.POSTERIOR);

        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 2, true, model);
        chunks.addListener(listener);

        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "3", "3", 2);
        addInstances(chunks, "0", "3", 2);
        addInstances(chunks, "3", "0", 2);

        chunks.addDistance(chunks.currentIndex);
        verify(listener).returnDriftPointMagnitude(chunks.currentIndex, new double[]{1.0});
    }

    @Test
    public void testCallBackJoint() {
        // Create Listeners
        TimelineListener listener = mock(TimelineListener.class);
        when(listener.getMeasurementType()).thenReturn(DriftMeasurement.JOINT);

        Model model = new FrequencyMaps(instances, 1, new int[]{0});
        Chunks chunks = new Chunks(0, 1, true, model);
        chunks.addListener(listener);

        addInstances(chunks, "0", "0", 2);
        addInstances(chunks, "3", "3", 2);

        chunks.addDistance(chunks.currentIndex);
        verify(listener).returnDriftPointMagnitude(chunks.currentIndex, new double[]{1.0});
    }
}