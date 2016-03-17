package main;

import main.generator.CategoricalDriftGenerator;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.opencsv.CSVWriter;
import com.yahoo.labs.samoa.instances.Instances;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.AbstractTask;
import moa.tasks.TaskMonitor;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by loongkuan on 24/02/16.
 **/
public class GenerateDataSet extends AbstractTask{
    public IntOption nAttributes = new IntOption("nAttributes", 'n',
	    "Number of attributes as parents of the class", 5, 1, 10);
    public IntOption nValuesPerAttribute = new IntOption("nValuesPerAttribute", 'v',
	    "Number of values per attribute", 3, 2, 5);
    public IntOption burnInNInstances = new IntOption("burnInNInstances", 'b',
	    "Number of instances before the start of the drift", 100000, 0, Integer.MAX_VALUE);
    public FloatOption driftMagnitudePrior = new FloatOption("driftMagnitudePrior", 'i',
	    "Magnitude of the drift between the starting probability and the one after the drift."
		    + " Magnitude is expressed as the Hellinger distance [0,1]", 0.5, 1e-20, 0.9);
    public FloatOption driftMagnitudeConditional = new FloatOption("driftMagnitudeConditional",
	    'o',
	    "Magnitude of the drift between the starting probability and the one after the drift."
		    + " Magnitude is expressed as the Hellinger distance [0,1]", 0.9, 1e-20, 0.9);
    public FloatOption precisionDriftMagnitude = new FloatOption(
	    "epsilon",
	    'e',
	    "Precision of the drift magnitude for p(x) (how far from the set magnitude is acceptable)",
	    0.01, 1e-20, 1.0);
    public FlagOption driftConditional = new FlagOption("driftConditional", 'c',
	    "States if the drift should apply to the conditional distribution p(y|x).");
    public FlagOption driftPriors = new FlagOption("driftPriors", 'p',
	    "States if the drift should apply to the prior distribution p(x). ");
    public IntOption seed = new IntOption("seed", 'r', "Seed for random number generator", -1,
	    Integer.MIN_VALUE, Integer.MAX_VALUE);
	public ClassOption baseGenerator = new ClassOption("generator", 'g', "The generator class to use to generate dataset",
			CategoricalDriftGenerator.class, "AbruptTreeDriftGenerator");
    public StringOption folderpath = new StringOption("folderpath", 'f', "Path to folder to store files in",
            "../datasets/");

    CategoricalDriftGenerator dataStream;

	@Override
    public void getDescription(StringBuilder sb, int indent) {

    }

    public Class<?> getTaskResultType() {
        return null;
    }

    @Override
    protected Object doTaskImpl(TaskMonitor monitor, ObjectRepository repository) {
        try {
            dataStream = (CategoricalDriftGenerator)getPreparedClassOption(this.baseGenerator);
            copyParameters(dataStream);

            Instances[] datasets = convertStreamToInstances();

            try {
                createDataset(datasets[0], folderpath.getValue(), "GeneratedDataBD.csv");
                createDataset(datasets[1], folderpath.getValue(), "GeneratedDataAD.csv");
                for (int i = 0; i < datasets[1].size(); i++) {
                    datasets[0].add(datasets[1].get(i));
                }
                createDataset(datasets[0], folderpath.getValue(), "GeneratedDataFull.csv");
            }
            catch (IOException ex){
                System.out.println("Error Creating Generator Instance");
            }
        }
        catch (Exception var2) {
            System.err.println("Creating a new classifier: " + var2.getMessage());
        }
        return CategoricalDriftGenerator.class;
    }

    private void copyParameters(CategoricalDriftGenerator dataStream) {
        dataStream.nAttributes.setValue(nAttributes.getValue());
        dataStream.nValuesPerAttribute.setValue(nValuesPerAttribute.getValue());
        dataStream.burnInNInstances.setValue(burnInNInstances.getValue());
        dataStream.driftMagnitudePrior.setValue(driftMagnitudePrior.getValue());
        dataStream.driftMagnitudeConditional.setValue(driftMagnitudeConditional.getValue());
        dataStream.precisionDriftMagnitude.setValue(precisionDriftMagnitude.getValue());
        dataStream.driftConditional.setValue(driftConditional.isSet());
        dataStream.driftPriors.setValue(driftPriors.isSet());
        dataStream.seed.setValue(seed.getValue());
    }

    private Instances[] convertStreamToInstances() {
        Instances[] convertedStreams = new Instances[2];

        dataStream.restart();
        dataStream.prepareForUse();
        convertedStreams[0] = new Instances(dataStream.getHeader(), dataStream.burnInNInstances.getValue());
        for (int i = 0; i < dataStream.burnInNInstances.getValue(); i++) {
            convertedStreams[0].add(dataStream.nextInstance().getData());
        }

        convertedStreams[1] = new Instances(dataStream.getHeader(), dataStream.burnInNInstances.getValue());
        for (int i = 0; i < dataStream.burnInNInstances.getValue(); i++) {
            convertedStreams[1].add(dataStream.nextInstance().getData());
        }

        return convertedStreams;
    }

    private void createDataset(Instances dataStream, String rootFolder, String filename) throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter(rootFolder + filename), '\t', CSVWriter.NO_QUOTE_CHARACTER);
        // feed in your array (or convert your data to an array)
        for (int i = 0; i < dataStream.size(); i++) {
            double[] inst = dataStream.instance(i).toDoubleArray();
            String[] dataToWrite = new String[inst.length];
            for (int j = 0; j < inst.length; j++) {
                dataToWrite[j] = Double.toString(inst[j]);
            }
            writer.writeNext(dataToWrite);
        }
        writer.close();
    }
}
