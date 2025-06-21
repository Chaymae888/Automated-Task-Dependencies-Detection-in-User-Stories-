### `g04-recycling.txt`(2017)
it concerns a web application where recycling and waste disposal facilities can be searched and located. The application operates through the visualization of a map that the user can interact with. The dataset has obtained from a [GitHub website](https://github.com/rafaellichen/Recycling-System/wiki/User-Stories) and it is at the basis of a students' project on web site design; the [code is available](https://github.com/rafaellichen/Recycling-System) (no license).
# As a user, I want to *"click"* on the *"address"*, so that it takes me to a *"new tab"* with *"Google Maps"*:

* Tasks: 

      1. Make address text clickable 
      2. Implement click handler to format address for Google Maps URL 
      3. Open Google Maps in new tab/window
      4. Add proper URL encoding for address parameters 
# As a user, I want to be able to *"anonymously"* *"view"* *"public information"*, so that I know about *"recycling centers"* *"near me"* *"before"* *"creating an account"*. 

* Tasks: 

1. Design public landing page layout 
2. Create anonymous user session handling 
3. Implement facility search without authentication 
4. Display basic facility information publicly 
5. Detect user’s location via browser API or IP. 
6. Show recycling centers within a radius of the user. 
7. Add "Sign up for more features" prompt
# As a user, I want to be able to *"enter "* my *"zip code"* and *"get a list of nearby recycling facilities"*, so that I can determine which ones I should consider. 
* Tasks: 
1. Design zip code input form 
2. Implement zip code validation (format and existence) 
3. Create geocoding service to convert zip to coordinates 
4. Build facility search algorithm by distance 
5. Design facility list display component 
6. Handle invalid zip code errors gracefully 
# As a user, I want to be able to get *"the hours of each recycling facility"*, so that I can *"arrange drop-offs"* on my *"off days"* or *"during after-work hours"*. 
* Tasks: 
1. Design hours display component (days/times) 
2. Create database schema for facility operating hours 
3. Implement hours parsing and validation 
4. Add special hours handling (holidays, closures) 
5. Display current open/closed status 
6. Add "hours today" quick view 
7. Handle timezone differences 
# As a user, I want to have a *"flexible"* *"pick up time"*, so that I can more conveniently use the website. 
* Tasks: 
1. Design pickup time selection interface 
2. Create time slot availability system 
3. Implement calendar/date picker component 
4. Add time validation logic 
5. Store user pickup preferences 
6. Send confirmation notifications 
7. Allow pickup time modifications 

# As a user, I want to be able to *"select different types of recyclable waste"*, so I have and *"get a list"* of facilities that *" accept each type"* and their *"opening hours"*, so that I can find an *"optimal route"* and *"schedule"*. 
- Tasks: 
1. Create waste type selection interface 
2. Build waste type database and categorization 
3. Implement facility filtering by accepted materials 
4. Create route optimization algorithm 
5. Display optimized route on map 
6. Show time estimates for each stop 
7. Export route to external navigation apps(Google Maps) 
# As a user, I want to *" add donation centers"* as *"favorites"* on *"my profile"*, so that I can view them later. 
- Tasks: 
1. Design profil interface 
2. Add favorite/bookmark button to facility cards 
3. Create favorites database table 
4. Implement add/remove favorite functionality 
5. Design favorites list page 
6. Add favorites management (organize, notes) 

# As a user, I want to be able to *"give my email ID"*, so that I can *"receive notifications"* for *"new events "* as they are posted. 
- Tasks: 
1. Create email subscription form 
2. Implement email validation and verification 
3. Build notification preferences interface 
4. Implement event notification triggers 
5. Add unsubscribe functionality 
6. Track email delivery status 

# As a user, I want to be able to *"view a map display"* of the public recycling bins around my area. 
- Tasks: 
1. Integrate mapping library (Google Maps/Mapbox) 
2. Create recycling bin database schema 
3. Design bin location markers and icons 
4. Implement area-based bin loading 
5. Add bin information popup/modal 
6. Filter bins by type and status 
7. Add user location centering
# As a user, I want to be able to *"view a map display "* of the *"special waste drop off sites"* *"around my area"*. 
- Tasks: 
1. Create special waste site data model 
2. Design distinct markers for special waste sites 
3. Add site information and accepted materials
4. Implement site filtering by waste type 
5. Show site availability and schedules 
6. Add driving directions integration 
7. Display site capacity information 
# As a user, I want to be able to *"view"* the *"safe disposal events"* *"currently"* being organised around *"my area"*. 
- Tasks: 
1. Create events database schema 
2. Design event listing interface 
3. Implement location-based event filtering 
4. Add event details (date, time, materials accepted) 
5. Create event RSVP functionality 
6. Add calendar integration 
7. Send event reminders 
# As a user, I want to *" view user documentation "* for the website, so that I know how to use the web app. 
- Tasks: 
1. Create help/documentation page structure 
2. Write user guides and tutorials 
3. Design searchable FAQ system 
4. Add contextual help tooltips 
5. Implement feedback system for documentation 
# As a user, I want to *"get feedback"* when I enter an *"invalid zip code"*. 
- Tasks: 
1. Implement real-time zip code validation 
2. Design error message components 
3. Add zip code format checking
4. Verify zip code exists in postal database 
5. Provide suggested corrections 
6. Add visual validation indicators 
7. Clear errors when valid input entered 
# As a user, I want to be able to *"create an account"*, so that I can create my own *" profile"*. 
- Tasks: 
1. Design user registration form 
2. Implement form validation (email, password strength) 
3. Create user database schema 
4. Add email verification system 
5. Build profile creation workflow 
6. Implement password hashing 
7. Add account activation process 
8. design user profile 
# As an admin, I want to be able to *"add or remove recycling facilities' information"*, so that users get the most recent information." 
- Tasks: 
1. Create admin facility management interface 
2. Build facility CRUD operations 
3. Design facility information form 
4. Implement facility data validation
5. Add bulk import/export functionality 
6. Create facility approval workflow 
7. Add audit trail for changes 
# As an admin, I want to be able to *"read users' feedback and complaints"*, so that we can add more features and keep improving the service we provide to them. 
- Tasks: 
1. Create feedback submission form 
2. Design admin feedback dashboard 
3. Implement feedback categorization 
4. Add feedback status tracking 
5. Create response system 
6. Generate feedback reports 
# As a user, I want to be able to *"check transaction history"* and keep a *"record"* of it, so that I can go back when needed. 
- Tasks: 
1. Create transaction logging system 
2. Design transaction history interface 
3. Add transaction search and filtering 
4. Implement transaction export functionality 
5. Create transaction details view 
6. Add transaction receipts/confirmation 
# As a user, I want to have a *"great UI and UX"* from the sites, so that I have a pleasant experience when navigating through them. 
- Tasks: 
1. Create design system and style guide 
2. Implement responsive design principles 
3. Design intuitive navigation structure 
4. Add smooth animations and transitions 
5. Optimize loading performance 6. Conduct usability testing 
6. Implement accessibility standards 
# As a user, I want to be able to *"access the site"* and do all the other stuffs *"on all of my electronic devices"*.
- Tasks: 
1. Implement responsive web design 
2. Test on various screen sizes
3. Optimize touch interactions for mobile 
4. Create Progressive Web App (PWA)
5. Test cross-browser compatibility 
6. Optimize performance for mobile networks 
7. Add offline functionality 
# As an admin, I want to be able to *"block specific users"* *"based on IP address"*, so that I can prevent spamming on the websites. 
- Tasks: 
1. Create IP blocking interface 
2. Implement IP address tracking 
3. Build blocked IP database 
4. Add IP range blocking capability
5. Create automatic spam detection 
6. Add temporary vs permanent blocking 
7. Implement IP whitelist functionality 
# As an admin, I want to view a *"dashboard"* that *"monitors all the sites' statuses"*, so that I can have a sense of what people are doing on our sites and know the service status. 
- Tasks: 
1. Design admin dashboard layout 
2. Create real-time analytics system 
3. Implement system health monitoring 
4. Add user activity tracking 
5. Create performance metrics display 
6. Build alert system for issues 
7. Add customizable dashboard widgets 
# As an admin, I want to have *"all data encrypted"*, so that important information will not be stolen during a server breach or an attack. 
- Tasks: 
1. Implement database encryption at rest 
2. Add SSL/TLS for data in transit 
3. Encrypt sensitive user data fields
4. Create secure key management system 
5. Implement data backup encryption 
6. Add security audit logging 
7. Regular security vulnerability scanning 
# As an executive, I want to have *" full access to data related to my company"*, so that I can have a sense of my company's performance. 
- Tasks: 
1. Create executive dashboard interface 
2. Implement role-based data access 
3. Build comprehensive analytics reports 
4. Add data visualization components 
5. Create export functionality for reports 
6. Implement real-time KPI tracking
7. Add comparative analysis tools 
# As an employee, I want to *"access"* the *"route planning system"* during work, so that I can be guided through the neighbourhood. 
- Tasks: 
1. Create employee portal interface 
2. Build route planning algorithm 
3. Integrate GPS navigation 
4. Add route optimization features 
5. Create mobile-friendly route display 
6. Implement turn-by-turn directions 
7. Add route tracking and completion
# As an employee from the HR department, I want to *"have access"* to the *" full information of all employees"* working for this business.
- Tasks: 
1. Create HR portal interface 
2. Implement employee database access 
3. Add employee information management 
4. Create employee search and filtering 
5. Implement HR reporting tools 
6. Add employee performance tracking
7. Ensure data privacy compliance 
# As a developer, I want to *"access"* an *"API"* from the website, so that I can integrate it and implement certain features in my own iOS application. 
- Tasks: 
1. Design RESTful API architecture 
2. Create API documentation 
3. Implement authentication for API access 
4. Add rate limiting and usage tracking 
5. Create API key management system
6. Add API versioning support
# As a user, I want to be able to *"receive tempting rewards"*, so that I have a reason to use the website. 
- Tasks: 
1. Design rewards/points system 
2. Create reward earning mechanics 
3. Build reward redemption interface 
4. Add gamification elements 
5. Create reward catalog 
6. Implement achievement badges 
7. Add reward notification system 
- # As a user, I want to have my *" personal information"* kept *"securely"* in the *"database"* of the website, so that I will not suffer from identity theft or telephone harassment. 
- Tasks: 
1. Implement data encryption for PII 
2. Create privacy settings interface 
3. Add data access controls
4. Implement secure data deletion 
5. Create privacy policy compliance
6. Add consent management 
7. Regular security audits 
- # As an admin, I want to *"handle all users' activities"*, so that I can manage more efficiently. 
- Tasks: 
1. Create user activity monitoring dashboard 
2. Implement user action logging 
3. Add user management tools 
4. Create bulk user operations 
5. Build user communication system
6. Add user behavior analytics 
7. Implement automated moderation rules 
- # As a company, I want to have a *"website"* that is *"easy to use"*, so that I can *"upload or delete"* stuff *"step by step"*. 
- Tasks: 
1. Create intuitive content management interface 
2. Design step-by-step upload wizard 
3. Implement drag-and-drop functionality 
4. Add bulk upload/delete operations 
5. Create file management system 
6. Add progress indicators 
7. Implement undo/redo functionality 
- # As an employee, I want to get *"quick notifications"*, so that I can process cases the first time. 
- Tasks: 
1. Create real-time notification system 
2. Implement push notifications 
3. Add notification preferences 
4. Create notification dashboard 
5. Add notification prioritization 
6. Implement notification acknowledgment
7. Add notification history 
- # As a company accountant, I want to *"view all available activity fees"* *"online"*, so that I can easily *"create a bill statement"*. 
- Tasks: 
1. Create fee management interface 
2. Build billing calculation system 
3. Add fee reporting tools 
4. Create bill generation system
5. Implement fee tracking 
6. Add payment integration 
7. Create financial reporting dashboard 
- # As a developer, I want to use *"bootstrap"* in the process of developing, so that
- I can easily design my website. 
- Tasks: 
1. Integrate Bootstrap framework 
2. Create custom Bootstrap theme 
3. Build component library 
4. Add responsive utilities 
5. Create design documentation 
6. Implement Bootstrap customization 
7. Add build process for Bootstrap 
- # As a developer, I want to *"attend some UI/UX lessons"*, so that I can develop an awesome and beautiful features website. 
- Tasks: 
1. Research UI/UX best practices 
2. Create design guidelines document 
3. Implement modern design patterns 
4. Add user testing protocols 
5. Create design review process 
6. Build design system 
7. Add accessibility standard
# As a user, I want to *"view all locations of recycling centers on a map"*, so that I can check which routes to take to drop off waste. 
- Tasks: 
1. Display all recycling centers on map 
2. Add center information markers 
3. Implement route calculation 
4. Add driving directions 
5. Show traffic information 
6. Create route comparison 
7. Add route sharing functionality 
# As a user, I want to *"upload my week's schedule"*, so that I can *""get recommendations""* for recycling centers that best *"fit my availability"*. 
- Tasks: 
1. Create schedule upload interface
2. Build schedule parsing system
3. Implement availability matching algorithm 
4. Create recommendation engine 
5. Add schedule conflict detection 
6. Build schedule management tools 
7. Add calendar integration 
# As a user, I want to *"link my email account to my profile"*, so that I can *"get a temporary password in case I forget my own one"*. - Tasks: 
1. Add email linking interface 
2. Implement email verification 
3. Create password reset system 
4. Build temporary password generation 
5. Add security questions option 
6. Implement password reset expiration 
7. Add account recovery logging
# As a user, I want to *"contact the administrators"*, so that I can *"give feedback"* or *"ask for help"*. 
- Tasks:
1. Create contact form interface
2. Implement message categorization 
3. Add file attachment support 
4. Create admin message dashboard 
5. Build response system 
6. Add message tracking 
7. Create FAQ integration 
# As an admin, I want to *"add recycling center information"*, so that I can keep the *"database"* *" up-to-date"* over time. 
- Tasks: 
1. Create center information form 
2. Implement data validation 
3. Add photo upload functionality 
4. Create center verification process 
5. Build data import tools 
6. Add change tracking 
7. Create approval workflow 
# As an admin, I want to *"view user error logs"*, so that I can *"fix "* or *"review any issues"* that are being faced by users of the system.
- Tasks: 
1. Implement error logging system 
2. Create error dashboard interface 
3. Add error categorization 
4. Build error analysis tools 
5. Create error notification system 
6. Add error resolution tracking 
7. Implement automated error reporting 

# As an admin, I want to *"onboard recycling centers "* on the platform, so that I can increase information accuracy. 
- Tasks: 
1. Create center onboarding workflow
2. Build center registration form 
3. Implement verification process 
4. Add documentation requirements 
5. Create approval system 
6. Build center training materials 
7. Add onboarding progress tracking 

# As a superuser, I want to *"update the recycling center information"*, so that I can *"provide the latest information about the recycling center"*. 
- Tasks: 
1. Create superuser interface 
2. Implement center information editing 
3. Add real-time updates 
4. Create change approval system 
5. Build version control 
6. Add bulk update tools 
7. Create update notifications 
# As a superuser, I want to *"view users' stats"*, so that I can view in real-time how many users have visited my recycling center information and their recyclable waste. 
- Tasks: 
1. Create analytics dashboard for centers 
2. Implement visitor tracking 
3. Add waste type analytics 
4. Create real-time statistics 
5. Build reporting tools
6. Add data export functionality 
7. Create trend analysis 
# As a superuser, I want to *"reply to user questions"*, so that I can answer any questions about my recycling center. 
- Tasks: 
1. Create Q&A interface 
2. Implement notification system for questions 
3. Add response management 
4. Create FAQ building tools 
5. Build question categorization 
6. Add response templates 
7. Implement rating system for answers 
1# As an admin, I want to be able to have a *"dashboard"* that shows *"usage stats"* and *"locations"*, so that I can identify the neighbourhoods with the largest number of drop-offs and to try getting more facilities involved. 
- Tasks: 
1. Create geographical analytics dashboard 
2. Implement location-based statistics 
3. Add heat map visualization 
4. Create demographic analysis 
5. Build trend identification 
6. Add facility gap analysis 
7. Create expansion recommendations 
# As an admin, I want to be able to *"communicate directly with facilities"*, so that I can keep them updated about features we have on our website.
- Tasks: 
1. Create facility communication portal 
2. Build messaging system 
3. Add broadcast functionality 
4. Create update notifications 
5. Implement communication templates 
6. Add facility response tracking 
7. Build communication analytics 
# As a user, I want to be able to *"browse through the list of facilities"* and *"see which ones are environment-friendly"*, so that I can know for sure my waste is not going to leave a negative ecological footprint. 
- Tasks: 
1. Create environmental rating system 
2. Add eco-friendly certification display 
3. Implement facility sustainability metrics 
4. Create environmental impact calculator 
5. Add green facility filtering 
6. Build sustainability reporting 
7. Create eco-friendly badges 
# As a recycling facility representative, I want to be able to *"update my information"* and *"the type of material I accept"*, so that I can avoid any miscommunication with users. 
- Tasks: 
1. Create facility portal interface 
2. Build information management system 
3. Add material acceptance updates 
4. Implement real-time information sync 
5. Create change notification system
6. Add validation for information accuracy
7. Build update approval workflow 
# As a recycling facility representative, I want to *"have access to user stats and schedules"*, so that I can adjust my hours and/or upgrade equipment and capacity in order to be able to accomodate larger amounts of recyclable materials. 
- Tasks: 
1. Create facility analytics dashboard 
2. Implement user visit predictions 
3. Add capacity planning tools 
4. Create demand forecasting 
5. Build equipment utilization tracking 
6. Add peak time analysis 
7. Create capacity alerts 
# As a recyclingfacility, I want to be able to *"communicate directly with the site admin"* and *"convey any issues or concerns"* I have, so that they fix them. 
- Tasks: 
1. Create facility support portal 
2. Build issue reporting system 
3. Add priority classification 
4. Create ticket tracking system 
5. Implement admin response system 
6. Add issue resolution tracking 
7. Create feedback loop for resolutions