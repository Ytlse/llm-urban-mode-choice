
FROM eclipse-temurin:21-jre-jammy
 
WORKDIR /otp
 
# Copy of the OTP jar (from bin/)
COPY bin/*.jar otp.jar
 
# Routing data (pre-built graph + configs)
COPY toulouse/ /var/otp/toulouse/
 
EXPOSE 8080
 
# --load: loads the existing graph.obj without rebuild
CMD ["java", "-Xmx4G", "-jar", "otp.jar", \
     "--load", "/var/otp/toulouse", \
     "--port", "8080"]
