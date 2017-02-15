package main.java.com.ips.kradziko.phd.core;

/**
 * Created by kradziko on 17/02/15.
 */
public class Model {
    private static Model model = new Model();

    private Model(){}
    public static Model getSingleton() { return model; }

}
